/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_dicom_h_
#define _itk_dicom_h_

#include "itk_image.h"

UCharImageType::Pointer load_dicom_uchar (char *dicom_dir);
ShortImageType::Pointer load_dicom_short (char *dicom_dir);
UShortImageType::Pointer load_dicom_ushort (char *dicom_dir);
FloatImageType::Pointer load_dicom_float (char *dicom_dir);

void save_image_dicom (ShortImageType::Pointer short_img, char* dir_name);

#endif
