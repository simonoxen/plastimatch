/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_file_format_h_
#define _plm_file_format_h_

#include "plm_config.h"

enum Plm_file_format {
    PLM_FILE_FMT_NO_FILE,
    PLM_FILE_FMT_UNKNOWN,
    PLM_FILE_FMT_IMG,
    PLM_FILE_FMT_VF,
    PLM_FILE_FMT_DIJ,
    PLM_FILE_FMT_POINTSET,
    PLM_FILE_FMT_CXT,
    PLM_FILE_FMT_DICOM_DIR,
    PLM_FILE_FMT_XIO_DIR,
    PLM_FILE_FMT_RTOG_DIR,
    PLM_FILE_FMT_PROJ_IMG,
    PLM_FILE_FMT_DICOM_RTSS,
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
