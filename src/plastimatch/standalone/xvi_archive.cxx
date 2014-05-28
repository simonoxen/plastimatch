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
#include "string_util.h"
#include "xvi_archive.h"

#if _WIN32
#define DEFAULT_DATABASE_DIRECTORY "D:/db"
#define DEFAULT_PATIENT_DIRECTORY "D:/db"
#else
#define DEFAULT_DATABASE_DIRECTORY ""
#define DEFAULT_PATIENT_DIRECTORY ""
#endif

#if defined (commentout)
int 
get_column_idx (
    P_DBF *p_dbf, 
    const char* column_name)
{
    int columns = dbf_NumCols(p_dbf);
    
    for (int i = 0; i < columns; i++) {
        const char* field_name = dbf_ColumnName(p_dbf, i);
        if (!strcmp (field_name, column_name)) {
            return i;
        }
    }
    return -1;
}

void
do_xvi_archive___old (Xvi_archive_parms *parms)
{
    int rc;
    P_DBF *p_dbf;

    std::string patient_dbf_fn = compose_filename (
        parms->database_dir, "PATIENT.DBF");

    /* Open the patient database */
    if (NULL == (p_dbf = dbf_Open (patient_dbf_fn.c_str()))) {
        print_and_exit ("Could not open dBASE file '%s'.\n", 
            patient_dbf_fn.c_str());
    }

    /* Get basic field info */
    int record_length = dbf_RecordLength(p_dbf);
    int num_records = dbf_NumRows (p_dbf);
    printf ("Found %d records of length %d.\n", num_records, record_length);

    int dbid_column_idx = get_column_idx (p_dbf, "DBID");
    int id_column_idx = get_column_idx (p_dbf, "ID");
    int dbid_len = dbf_ColumnSize (p_dbf, dbid_column_idx);
    int id_len = dbf_ColumnSize (p_dbf, id_column_idx);
    printf ("DBID: Column = %d, Len = %d\n", dbid_column_idx, dbid_len);
    printf ("ID: Column = %d, Len = %d\n", id_column_idx, id_len);

    /* Is this needed? */
    int start_record = 1;
    rc = dbf_SetRecordOffset (p_dbf, start_record);
    if (rc < 0) {
        print_and_exit ("Can't set start record.\n");
    }

    /* Loop through rows */
    char *record = new char[record_length + 1];
    char *id = new char[dbf_ColumnSize(p_dbf,id_column_idx) + 1];
    char *dbid = new char[dbf_ColumnSize(p_dbf,dbid_column_idx) + 1];
    char *record_data;
    for (int i = 1; i <= num_records; i++) {
        rc = dbf_ReadRecord (p_dbf, (char*) record, record_length);
        if (rc < 0) {
            printf ("Error reading record %d (rc = %d)\n", i, rc);
            break;
        }
        record_data = dbf_GetRecordData (p_dbf, record, id_column_idx);
        memcpy (id, record_data, id_len);
        id[id_len] = 0;

        if (0 == strncmp (record_data, parms->patient_id.c_str(), strlen(parms->patient_id.c_str()))) {
            record_data = dbf_GetRecordData (p_dbf, record, dbid_column_idx);
            memcpy (dbid, record_data, dbid_len);
            dbid[dbid_len] = 0;
            printf ("ID %s -> DBID %s\n", id, dbid);
        }
    }
    delete record;
    delete id;
    delete dbid;

    dbf_Close(p_dbf);
}
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
        printf ("Name = %s^%s\n", 
            recon_ini.Get ("IDENTIFICATION", "LastName", "").c_str(),
            recon_ini.Get ("IDENTIFICATION", "FirstName", "").c_str());
    }

#if defined (commentout)
    int rc;
    P_DBF *p_dbf;

    std::string patient_dbf_fn = compose_filename (
        parms->database_dir, "PATIENT.DBF");

    /* Open the patient database */
    if (NULL == (p_dbf = dbf_Open (patient_dbf_fn.c_str()))) {
        print_and_exit ("Could not open dBASE file '%s'.\n", 
            patient_dbf_fn.c_str());
    }

    /* Get basic field info */
    int record_length = dbf_RecordLength(p_dbf);
    int num_records = dbf_NumRows (p_dbf);
    printf ("Found %d records of length %d.\n", num_records, record_length);

    int dbid_column_idx = get_column_idx (p_dbf, "DBID");
    int id_column_idx = get_column_idx (p_dbf, "ID");
    int dbid_len = dbf_ColumnSize (p_dbf, dbid_column_idx);
    int id_len = dbf_ColumnSize (p_dbf, id_column_idx);
    printf ("DBID: Column = %d, Len = %d\n", dbid_column_idx, dbid_len);
    printf ("ID: Column = %d, Len = %d\n", id_column_idx, id_len);

    /* Is this needed? */
    int start_record = 1;
    rc = dbf_SetRecordOffset (p_dbf, start_record);
    if (rc < 0) {
        print_and_exit ("Can't set start record.\n");
    }

    /* Loop through rows */
    char *record = new char[record_length + 1];
    char *id = new char[dbf_ColumnSize(p_dbf,id_column_idx) + 1];
    char *dbid = new char[dbf_ColumnSize(p_dbf,dbid_column_idx) + 1];
    char *record_data;
    for (int i = 1; i <= num_records; i++) {
        rc = dbf_ReadRecord (p_dbf, (char*) record, record_length);
        if (rc < 0) {
            printf ("Error reading record %d (rc = %d)\n", i, rc);
            break;
        }
        record_data = dbf_GetRecordData (p_dbf, record, id_column_idx);
        memcpy (id, record_data, id_len);
        id[id_len] = 0;

        if (0 == strncmp (record_data, parms->patient_id.c_str(), strlen(parms->patient_id.c_str()))) {
            record_data = dbf_GetRecordData (p_dbf, record, dbid_column_idx);
            memcpy (dbid, record_data, dbid_len);
            dbid[dbid_len] = 0;
            printf ("ID %s -> DBID %s\n", id, dbid);
        }
    }
    delete record;
    delete id;
    delete dbid;

    dbf_Close(p_dbf);
#endif
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
