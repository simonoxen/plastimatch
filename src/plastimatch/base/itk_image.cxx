/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkOrientImageFilter.h"

#include "plmbase.h"
#include "plmsys.h"

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
itk_image_set_header (T dest, const Plm_image_header *pih)
{
    dest->SetRegions (pih->m_region);
    dest->SetOrigin (pih->m_origin);
    dest->SetSpacing (pih->m_spacing);
    dest->SetDirection (pih->m_direction);
}

template<class T>
void
itk_image_set_header (T dest, const Plm_image_header& pih)
{
    itk_image_set_header (dest, &pih);
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

/* Return true if the headers are the same */
template<class T, class U>
bool 
itk_image_header_compare (T image1, U image2)
{
    typedef typename U::ObjectType I1ImageType;
    typedef typename T::ObjectType I2ImageType;

    const typename I1ImageType::SizeType& i1_sz
	= image1->GetLargestPossibleRegion().GetSize ();
    const typename I1ImageType::PointType& i1_og = image1->GetOrigin();
    const typename I1ImageType::SpacingType& i1_sp = image1->GetSpacing();
    const typename I1ImageType::DirectionType& i1_dc = image1->GetDirection();

    const typename I2ImageType::SizeType& i2_sz
	= image2->GetLargestPossibleRegion().GetSize ();
    const typename I2ImageType::PointType& i2_og = image2->GetOrigin();
    const typename I2ImageType::SpacingType& i2_sp = image2->GetSpacing();
    const typename I2ImageType::DirectionType& i2_dc = image2->GetDirection();

    if (i1_sz != i2_sz || i1_og != i2_og || i1_sp != i2_sp || i1_dc != i2_dc)
    {
        return false;
    } else {
        return true;
    }
}

template<class T> 
void 
itk_volume_center (float center[3], const T image)
{
    Itk_volume_header ivh;
    ivh.set_from_itk_image (image);
    ivh.get_image_center (center);
}

/* Explicit instantiations */
template PLMBASE_API void get_image_header (plm_long dim[3], float offset[3], float spacing[3], UCharImageType::Pointer image);
template PLMBASE_API void get_image_header (plm_long dim[3], float offset[3], float spacing[3], ShortImageType::Pointer image);
template PLMBASE_API void get_image_header (plm_long dim[3], float offset[3], float spacing[3], UShortImageType::Pointer image);
template PLMBASE_API void get_image_header (plm_long dim[3], float offset[3], float spacing[3], FloatImageType::Pointer image);
template PLMBASE_API void itk_image_set_header (UCharVecImageType::Pointer, const Plm_image_header *pih);
template PLMBASE_API void itk_image_set_header (DeformationFieldType::Pointer, const Plm_image_header *pih);
template PLMBASE_API void itk_image_set_header (UCharVecImageType::Pointer, const Plm_image_header& pih);
template PLMBASE_API void itk_image_set_header (DeformationFieldType::Pointer, const Plm_image_header& pih);
template PLMBASE_API void itk_image_header_copy (UCharVecImageType::Pointer, UCharImageType::Pointer);
template PLMBASE_API void itk_image_header_copy (UCharVecImageType::Pointer, UInt32ImageType::Pointer);
template PLMBASE_API void itk_image_header_copy (UCharVecImageType::Pointer, UCharVecImageType::Pointer);
template PLMBASE_API void itk_image_header_copy (UCharVecImageType::Pointer, DeformationFieldType::Pointer);
template PLMBASE_API void itk_image_header_copy (UCharImageType::Pointer, UCharVecImageType::Pointer);
template PLMBASE_API void itk_image_header_copy (UCharImageType::Pointer, FloatImageType::Pointer);
template PLMBASE_API void itk_image_header_copy (UCharImage2DType::Pointer, UCharVecImage2DType::Pointer);
template PLMBASE_API void itk_image_get_volume_header (Volume_header *, DeformationFieldType::Pointer);
template PLMBASE_API bool itk_image_header_compare (UCharImageType::Pointer image1, UCharImageType::Pointer image2);
template PLMBASE_API void itk_volume_center (float center[3], const FloatImageType::Pointer image);

