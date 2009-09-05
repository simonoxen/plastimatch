/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include "plm_config.h"

File_type
deduce_file_type (char* path)
{
    return FILE_TYPE_IMG;
}

char*
file_type_string (File_type file_type)
{
    switch (file_type) {
    case FILE_TYPE_UNKNOWN:
	return "Unknown";
	break;
    case FILE_TYPE_IMG:
	return "Image";
	break;
    case FILE_TYPE_DIJ:
	return "Dij matrix";
	break;
    case FILE_TYPE_POINTSET:
	return "Pointset";
	break;
    case FILE_TYPE_CXT:
	return "Cxt file";
	break;
    case FILE_TYPE_DICOM_DIR:
	return "Dicom directory";
	break;
    case FILE_TYPE_XIO_DIR:
	return "XiO directory";
	break;
    case FILE_TYPE_RTOG_DIR:
	return "RTOG directory";
	break;
    default:
	return "Unknown/default";
	break;
    }
}
