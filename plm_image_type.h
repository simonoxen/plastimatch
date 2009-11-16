/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_image_type_h_
#define _plm_image_type_h_

/* We only deal with these kinds of images. */
enum PlmImageType {
    PLM_IMG_TYPE_UNDEFINED, 
    PLM_IMG_TYPE_ITK_CHAR, 
    PLM_IMG_TYPE_ITK_UCHAR, 
    PLM_IMG_TYPE_ITK_SHORT, 
    PLM_IMG_TYPE_ITK_USHORT, 
    PLM_IMG_TYPE_ITK_LONG, 
    PLM_IMG_TYPE_ITK_ULONG, 
    PLM_IMG_TYPE_ITK_FLOAT, 
    PLM_IMG_TYPE_ITK_DOUBLE, 
    PLM_IMG_TYPE_ITK_FLOAT_FIELD, 
    PLM_IMG_TYPE_GPUIT_FLOAT, 
    PLM_IMG_TYPE_GPUIT_FLOAT_FIELD, 
};

PlmImageType
plm_image_type_parse (const char* string);

#endif
