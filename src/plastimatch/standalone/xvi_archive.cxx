/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
   Algorithm for doing an XVI archive.
   
   ** This algorithm does not use dbase file, which reduces need
   to access xvi computer **

   (1) Get patient ID (PATID) from command line argument
   (2) Search through file system, identify CBCT images according to 
       patient_<PATID>/IMAGES/img_<CBCT_UID>/Reconstruction/<RECON_UID>.SCAN
   (3) Load Reconstruction/<RECON_UID>.INI
   (3a) Parse this file to get Name, TreatmentID, 
   (4) Load Reconstruction/<RECON_UID>.INI.XVI
   (4a) Parse this file to get Date, Xform
   (5) Identify reference image in Mosaiq (optional?)
   

   ----------------------------------------------------------------------- */
#include <stdlib.h>
#include <string.h>
#include "INIReader.h"

#include "plm_config.h"
#include "dcmtk_sro.h"
#include "dir_list.h"
#include "file_util.h"
#include "path_util.h"
#include "plm_clp.h"
#include "print_and_exit.h"
#include "rt_study.h"
#include "string_util.h"
#include "xvi_archive.h"

#if _WIN32
#define DEFAULT_DATABASE_DIRECTORY "D:/db"
#define DEFAULT_PATIENT_DIRECTORY "D:/db"
#else
#define DEFAULT_DATABASE_DIRECTORY ""
#define DEFAULT_PATIENT_DIRECTORY ""
#endif

void
do_xvi_archive (Xvi_archive_parms *parms)
{
    std::string patient_dir = compose_filename (
        parms->patient_base_dir, 
        string_format ("patient_%s", parms->patient_id.c_str()));
    std::string patient_ct_set_dir = compose_filename (
        patient_dir, "CT_SET");
    std::string patient_images_dir = compose_filename (
        patient_dir, "IMAGES");

    /* Load one of the reference CTs */
    Dir_list ct_set_dir (patient_ct_set_dir);
    if (ct_set_dir.num_entries == 0) {
        printf ("Error.  No CT_SET found.\n");
        return;
    }
    Rt_study reference_study;
    std::string reference_uid;
    for (int i = 0; i < ct_set_dir.num_entries; i++) {
        if (ct_set_dir.entries[i][0] == '.') {
            continue;
        }
        std::string fn = string_format ("%s/%s",
            patient_ct_set_dir.c_str(), ct_set_dir.entries[i]);
        if (is_directory (fn)) {
            printf ("Loaded reference study (%s)\n", fn.c_str());
            reference_study.load_image (fn);
            reference_uid = ct_set_dir.entries[i];
            break;
        }
    }
    if (!reference_study.have_image()) {
        printf ("Error.  No reference CT loaded.\n");
        return;
    }
    Rt_study_metadata::Pointer& reference_meta = 
        reference_study.get_rt_study_metadata ();
    printf ("Reference Meta: %s %s\n",
        reference_meta->get_patient_name().c_str(),
        reference_meta->get_patient_id().c_str());

    Dir_list images_dir (patient_images_dir);
    if (images_dir.num_entries == 0) {
        printf ("Error.  No images found.\n");
        return;
    }

    /* For potential CBCT image */
    for (int i = 0; i < images_dir.num_entries; i++) {
        if (images_dir.entries[i][0] == '.') {
            continue;
        }
        /* Find pertinent filenames */
        std::string recon_dir = string_format ("%s/%s/Reconstruction",
            patient_images_dir.c_str(), images_dir.entries[i]);
        Dir_list recon_list (recon_dir);
        std::string scan_fn, recon_ini_fn, recon_xvi_fn;
        for (int j = 0; j < recon_list.num_entries; j++) {
            if (!extension_is (recon_list.entries[j], "SCAN")) {
                continue;
            }
            scan_fn = compose_filename (recon_dir, recon_list.entries[j]);
            std::string recon_uid 
                = strip_extension (std::string(recon_list.entries[j]));
            recon_ini_fn = compose_filename (recon_dir, 
                string_format ("%s.INI", recon_uid.c_str()));
            recon_xvi_fn = compose_filename (recon_dir, 
                string_format ("%s.INI.XVI", recon_uid.c_str()));
            break;
        }
        if (scan_fn == ""
            || !file_exists (scan_fn) 
            || !file_exists (recon_ini_fn)
            || !file_exists (recon_xvi_fn))
        {
            continue;
        }

        /* Load the INI file */
        INIReader recon_ini (recon_ini_fn);
        std::string cbct_ref_uid = 
            recon_ini.Get ("IDENTIFICATION", "ReferenceUID", "");
        std::string patient_name = 
            string_format ("%s^%s",
                recon_ini.Get ("IDENTIFICATION", "LastName", "").c_str(),
                recon_ini.Get ("IDENTIFICATION", "FirstName", "").c_str());

        printf ("name = %s\n", patient_name.c_str());
        printf ("reference_uid = %s\n", cbct_ref_uid.c_str());

        /* Verify if the file belongs to this reference CT */
        if (cbct_ref_uid != reference_uid) {
            printf ("Reference UID mismatch.  Skipping.\n");
            continue;
        }

        /* Load the INI.XVI file */
        INIReader recon_xvi (recon_xvi_fn);
        std::string registration_string = 
            recon_xvi.Get ("ALIGNMENT", "OnlineToRefTransformCorrection", "");
        printf ("xform = %s\n", registration_string.c_str());

        /* Load the .SCAN */
        Rt_study cbct_study;
        cbct_study.load_image (scan_fn);

        if (!cbct_study.have_image()) {
            printf ("ERROR: decompression failure with patient %s\n",
                parms->patient_id.c_str());
            exit (1);
        }

        /* Write the DICOM image */
        std::string output_dir = string_format (
            "cbct_output/%s", images_dir.entries[i]);

        Rt_study_metadata::Pointer& cbct_meta 
            = cbct_study.get_rt_study_metadata ();
        if (parms->patient_id_override != "") {
            cbct_meta->set_patient_id (parms->patient_id_override);
        } else {
            cbct_meta->set_patient_id (parms->patient_id);
        }
        cbct_meta->set_patient_name (patient_name);
        cbct_study.save_dicom (output_dir);

        /* Create the DICOM SRO */
        Xform::Pointer xf = Xform::New();
        AffineTransformType::ParametersType xfp(12);
        float xvip[16];
        int rc = sscanf (registration_string.c_str(), 
            "%f %f %f %f %f %f %f %f "
            "%f %f %f %f %f %f %f %f",
            &xvip[0], &xvip[1], &xvip[2], &xvip[3], 
            &xvip[4], &xvip[5], &xvip[6], &xvip[7], 
            &xvip[8], &xvip[9], &xvip[10], &xvip[11], 
            &xvip[12], &xvip[13], &xvip[14], &xvip[15]);
        if (rc != 16) {
            printf ("Error parsing transform string.\n");
            exit (1);
        }
        xfp[0] = xvip[0];
        xfp[1] = xvip[1];
        xfp[2] = xvip[2];
        xfp[3] = xvip[4];
        xfp[4] = xvip[5];
        xfp[5] = xvip[6];
        xfp[6] = xvip[8];
        xfp[7] = xvip[9];
        xfp[8] = xvip[10];
        xfp[9] = xvip[12];
        xfp[10] = xvip[13];
        xfp[11] = xvip[14];
        xf->set_aff (xfp);

        Dcmtk_sro::save (
            xf,
            reference_study.get_rt_study_metadata (),
            cbct_study.get_rt_study_metadata (),
            output_dir);

        //break;
    }
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: xvi_archive [options]\n");
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Xvi_archive_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Input files and directories */
    parser->add_long_option ("", "patient-directory", 
        "base directory containing patient images", 1, 
        DEFAULT_PATIENT_DIRECTORY);

    /* Other options */
    parser->add_long_option ("", "patient-id-override", 
        "set the patient id", 1);
    
    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that a patient id was given */
    if (parser->number_of_arguments() == 0) {
	throw (dlib::error ("Error.  You must specify a patient ID"));
    }

    /* Input files */
    parms->patient_base_dir = parser->get_string("patient-directory");
    if (parms->patient_base_dir == "") {
        throw (dlib::error (
                "Error.  The use of --patient-directory is needed"));
    }

    /* Other options */
    parms->patient_id_override = parser->get_string("patient-id-override");

    parms->patient_id = (*parser)[0];
}


int
main(int argc, char *argv[])
{
    Xvi_archive_parms parms;

    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 0);

    /* Do the job */
    do_xvi_archive (&parms);
}
