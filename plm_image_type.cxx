/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <string.h>
#include "plm_image_type.h"

Plm_image_type
plm_image_type_parse (const char* string)
{
    if (!strcmp (string,"char")) {
	return PLM_IMG_TYPE_ITK_CHAR;
    }
    else if (!strcmp (string,"mask") || !strcmp (string,"uchar")) {
	return PLM_IMG_TYPE_ITK_UCHAR;
    }
    else if (!strcmp (string,"short")) {
	return PLM_IMG_TYPE_ITK_SHORT;
    }
    else if (!strcmp (string,"ushort")) {
	return PLM_IMG_TYPE_ITK_USHORT;
    }
    else if (!strcmp (string,"int") || !strcmp (string,"long")
	     || !strcmp (string,"int32")) {
	return PLM_IMG_TYPE_ITK_LONG;
    }
    else if (!strcmp (string,"uint") || !strcmp (string,"ulong")
	     || !strcmp (string,"uint32")) {
	return PLM_IMG_TYPE_ITK_ULONG;
    }
    else if (!strcmp (string,"float")) {
	return PLM_IMG_TYPE_ITK_FLOAT;
    }
    else if (!strcmp (string,"vf")) {
	return PLM_IMG_TYPE_ITK_FLOAT_FIELD;
    }
    else {
	return PLM_IMG_TYPE_UNDEFINED;
    }
}
