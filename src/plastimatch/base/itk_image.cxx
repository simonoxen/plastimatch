/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkOrientImageFilter.h"

#include "itk_dicom_load.h"
#include "itk_image.h"
#include "itk_image_cast.h"
#include "file_util.h"
#include "logfile.h"
#include "plm_image_header.h"
#include "plm_int.h"
#include "volume_header.h"

#if (defined(_WIN32) || defined(WIN32))
#define snprintf _snprintf
#define mkdir(a,b) _mkdir(a)
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

/* -----------------------------------------------------------------------
    Functions
   ----------------------------------------------------------------------- */
void
itk_image_get_props (
    const std::string& fileName,
    int *num_dimensions, 
    itk::ImageIOBase::IOPixelType *pixel_type, 
    itk::ImageIOBase::IOComponentType *component_type, 
    int *num_components
)
{
    *pixel_type = itk::ImageIOBase::UNKNOWNPIXELTYPE;
    *component_type = itk::ImageIOBase::UNKNOWNCOMPONENTTYPE;
    *num_dimensions = 0;
    *num_components = 0;
    typedef itk::Image<short, 3> ImageType;
    itk::ImageFileReader<ImageType>::Pointer imageReader =
	itk::ImageFileReader<ImageType>::New();
    imageReader->SetFileName(fileName.c_str());
    try {
	imageReader->UpdateOutputInformation();
	*pixel_type = imageReader->GetImageIO()->GetPixelType();
	*component_type = imageReader->GetImageIO()->GetComponentType();
	*num_dimensions = imageReader->GetImageIO()->GetNumberOfDimensions();
	*num_components = imageReader->GetImageIO()->GetNumberOfComponents();
    } catch (itk::ExceptionObject &ex) {
	printf ("ITK exception.\n");
	std::cout << ex << std::endl;
    }
}

/* -----------------------------------------------------------------------
   Reading Image Headers
   ----------------------------------------------------------------------- */
template<class T>
void
get_image_header (plm_long dim[3], float offset[3], float spacing[3], T image)
{
    typename T::ObjectType::RegionType rg = image->GetLargestPossibleRegion ();
    typename T::ObjectType::PointType og = image->GetOrigin();
    typename T::ObjectType::SpacingType sp = image->GetSpacing();
    typename T::ObjectType::SizeType sz = rg.GetSize();

    /* Copy header & allocate data for gpuit float */
    for (int d = 0; d < 3; d++) {
	dim[d] = sz[d];
	offset[d] = og[d];
	spacing[d] = sp[d];
    }
}

template<class T>
void
itk_image_get_image_header (plm_long dim[3], float offset[3], float spacing[3], 
    Direction_cosines& dc, const T image)
{
    typename T::ObjectType::RegionType rg = image->GetLargestPossibleRegion ();
    typename T::ObjectType::PointType og = image->GetOrigin();
    typename T::ObjectType::SpacingType sp = image->GetSpacing();
    typename T::ObjectType::SizeType sz = rg.GetSize();

    /* Copy header & allocate data for gpuit float */
    for (int d = 0; d < 3; d++) {
	dim[d] = sz[d];
	offset[d] = og[d];
	spacing[d] = sp[d];
    }

    /* Copy direction cosines */
    DirectionType itk_dc = image->GetDirection ();
    for (int a = 0; a < 3; a++) {
	for (int b = 0; b < 3; b++) {
	    dc[a*3+b] = itk_dc[a][b];
	}
    }
}

template<class T>
void
itk_image_get_volume_header (Volume_header *vh, T image)
{
    itk_image_get_image_header (vh->m_dim, vh->m_origin, vh->m_spacing, 
	vh->m_direction_cosines, image);
}

template<class T>
void
itk_image_set_header (T dest, Plm_image_header *pih)
{
    dest->SetRegions (pih->m_region);
    dest->SetOrigin (pih->m_origin);
    dest->SetSpacing (pih->m_spacing);
    dest->SetDirection (pih->m_direction);
}

template<class T, class U>
void
itk_image_header_copy (T dest, U src)
{
    typedef typename U::ObjectType SrcImageType;
    typedef typename T::ObjectType DestImageType;

    const typename SrcImageType::RegionType src_rgn
	= src->GetLargestPossibleRegion();
    const typename SrcImageType::PointType& src_og = src->GetOrigin();
    const typename SrcImageType::SpacingType& src_sp = src->GetSpacing();
    const typename SrcImageType::DirectionType& src_dc = src->GetDirection();

    dest->SetRegions (src_rgn);
    dest->SetOrigin (src_og);
    dest->SetSpacing (src_sp);
    dest->SetDirection (src_dc);
}

/* Explicit instantiations */
template plastimatch1_EXPORT void get_image_header (plm_long dim[3], float offset[3], float spacing[3], UCharImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (plm_long dim[3], float offset[3], float spacing[3], ShortImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (plm_long dim[3], float offset[3], float spacing[3], UShortImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (plm_long dim[3], float offset[3], float spacing[3], FloatImageType::Pointer image);
template plastimatch1_EXPORT void itk_image_set_header (UCharVecImageType::Pointer, Plm_image_header *pih);
template plastimatch1_EXPORT void itk_image_set_header (DeformationFieldType::Pointer, Plm_image_header *pih);
template plastimatch1_EXPORT void itk_image_header_copy (UCharVecImageType::Pointer, UCharImageType::Pointer);
template plastimatch1_EXPORT void itk_image_header_copy (UCharVecImageType::Pointer, UInt32ImageType::Pointer);
template plastimatch1_EXPORT void itk_image_header_copy (UCharVecImageType::Pointer, UCharVecImageType::Pointer);
template plastimatch1_EXPORT void itk_image_header_copy (UCharVecImageType::Pointer, DeformationFieldType::Pointer);
template plastimatch1_EXPORT void itk_image_header_copy (UCharImageType::Pointer, UCharVecImageType::Pointer);
template plastimatch1_EXPORT void itk_image_header_copy (UCharImage2DType::Pointer, UCharVecImage2DType::Pointer);
template plastimatch1_EXPORT void itk_image_get_volume_header (Volume_header *, DeformationFieldType::Pointer);
