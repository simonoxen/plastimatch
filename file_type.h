/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _file_type_h_
#define _file_type_h_

enum Plm_file_type {
    PLM_FILE_TYPE_NO_FILE,
    PLM_FILE_TYPE_UNKNOWN,
    PLM_FILE_TYPE_IMG,
    PLM_FILE_TYPE_VF,
    PLM_FILE_TYPE_DIJ,
    PLM_FILE_TYPE_POINTSET,
    PLM_FILE_TYPE_CXT,
    PLM_FILE_TYPE_DICOM_DIR,
    PLM_FILE_TYPE_XIO_DIR,
    PLM_FILE_TYPE_RTOG_DIR,
    PLM_FILE_TYPE_PFM,
};

Plm_file_type
deduce_file_type (char* path);
char*
file_type_string (Plm_file_type);

#endif
