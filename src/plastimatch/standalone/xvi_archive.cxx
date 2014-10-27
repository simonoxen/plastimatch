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

    Dir_list image_list (patient_images_dir);


    if (image_list.num_entries == 0) {
        printf ("Error.  No images found.\n");
    }

    /* For potential CBCT image */
    for (int i = 0; i < image_list.num_entries; i++) {
        if (image_list.entries[i][0] == '.') {
            continue;
        }
        /* Find pertinent filenames */
        std::string recon_dir = string_format ("%s/%s/Reconstruction",
            patient_images_dir.c_str(), image_list.entries[i]);
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
        printf ("name = %s^%s\n", 
            recon_ini.Get ("IDENTIFICATION", "LastName", "").c_str(),
            recon_ini.Get ("IDENTIFICATION", "FirstName", "").c_str());

        /* Load the INI.XVI file */
        INIReader recon_xvi (recon_xvi_fn);
        printf ("xform = %s\n", recon_xvi.Get ("ALIGNMENT", 
                "OnlineToRefTransformCorrection", "").c_str());

        /* Load the .SCAN */
        Rt_study rt_study;
        rt_study.load_image (scan_fn);

        if (!rt_study.have_image()) {
            printf ("ERROR: decompression failure with patient %s\n",
                parms->patient_id.c_str());
            exit (1);
        }

        /* Write the DICOM image */
        std::string output_dir = string_format (
            "cbct_output/%s", image_list.entries[i]);
        rt_study.save_dicom (output_dir);

        /* Create the DICOM SRO */

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
