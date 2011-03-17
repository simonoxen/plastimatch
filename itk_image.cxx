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

#include "plm_int.h"
#include "itk_dicom_load.h"
#include "itk_image.h"
#include "itk_image_cast.h"
#include "file_util.h"
#include "print_and_exit.h"
#include "logfile.h"
#include "plm_image_patient_position.h"

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
    std::string fileName,
    int *num_dimensions, 
    itk::ImageIOBase::IOPixelType &pixel_type, 
    itk::ImageIOBase::IOComponentType &component_type, 
    int *num_components
)
{
    pixel_type = itk::ImageIOBase::UNKNOWNPIXELTYPE;
    component_type = itk::ImageIOBase::UNKNOWNCOMPONENTTYPE;
    *num_dimensions = 0;
    typedef itk::Image<short, 3> ImageType;
    itk::ImageFileReader<ImageType>::Pointer imageReader =
	itk::ImageFileReader<ImageType>::New();
    imageReader->SetFileName(fileName.c_str());
    try {
	imageReader->UpdateOutputInformation();
	pixel_type = imageReader->GetImageIO()->GetPixelType();
	component_type = imageReader->GetImageIO()->GetComponentType();
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
get_image_header (int dim[3], float offset[3], float spacing[3], T image)
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

/* Explicit instantiations */
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], UCharImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], ShortImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], UShortImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], FloatImageType::Pointer image);
