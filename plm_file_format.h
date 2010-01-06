/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_file_format_h_
#define _plm_file_format_h_

#include "plm_config.h"

enum Plm_file_format {
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
    PLM_FILE_TYPE_PROJ_IMG,
    PLM_FILE_TYPE_DICOM_RTSS,
};

plastimatch1_EXPORT
Plm_file_format
plm_file_format_deduce (char* path);
plastimatch1_EXPORT
char*
plm_file_format_string (Plm_file_format file_type);
plastimatch1_EXPORT
Plm_file_format 
plm_file_format_parse (const char* string);

#endif
