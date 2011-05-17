/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_dicom_load_h_
#define _itk_dicom_load_h_

#include "plm_config.h"
#include "itk_image.h"
#include "plm_image_patient_position.h"

UCharImageType::Pointer load_dicom_uchar (const char *dicom_dir);
ShortImageType::Pointer load_dicom_short (const char *dicom_dir);
UShortImageType::Pointer load_dicom_ushort (const char *dicom_dir);
Int32ImageType::Pointer load_dicom_int32 (const char *dicom_dir);
UInt32ImageType::Pointer load_dicom_uint32 (const char *dicom_dir);
FloatImageType::Pointer load_dicom_float (const char *dicom_dir);
DoubleImageType::Pointer load_dicom_double (const char *dicom_dir);

#endif
