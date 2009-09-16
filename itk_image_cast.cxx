/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include <stdlib.h>
#include <string.h>
#if (defined(_WIN32) || defined(WIN32))
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#endif
#include "plm_config.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkOrientImageFilter.h"
#include "itk_image.h"
#include "print_and_exit.h"
#include "itk_dicom.h"
#include "logfile.h"

#if (defined(_WIN32) || defined(WIN32))
#define snprintf _snprintf
#define mkdir(a,b) _mkdir(a)
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

/* -----------------------------------------------------------------------
   Casting image types
   ----------------------------------------------------------------------- */
template<class T> 
UCharImageType::Pointer
cast_uchar (T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::CastImageFilter <
	ImageType, UCharImageType > CastFilterType;

    typename CastFilterType::Pointer caster = CastFilterType::New();
    caster->SetInput(image);
    try {
	caster->Update();
    }
    catch (itk::ExceptionObject & ex) {
	printf ("ITK exception in CastFilter.\n");
	std::cout << ex << std::endl;
	exit(1);
    }
    return caster->GetOutput();
}

template<class T> 
ShortImageType::Pointer
cast_short (T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::CastImageFilter <
	ImageType, ShortImageType > CastFilterType;

    typename CastFilterType::Pointer caster = CastFilterType::New();
    caster->SetInput(image);
    try {
	caster->Update();
    }
    catch (itk::ExceptionObject & ex) {
	printf ("ITK exception in CastFilter.\n");
	std::cout << ex << std::endl;
	exit(1);
    }
    return caster->GetOutput();
}

template<class T> 
FloatImageType::Pointer
cast_float (T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::CastImageFilter <
	ImageType, FloatImageType > CastFilterType;

    typename CastFilterType::Pointer caster = CastFilterType::New();
    caster->SetInput(image);
    try {
	caster->Update();
    }
    catch (itk::ExceptionObject & ex) {
	printf ("ITK exception in CastFilter.\n");
	std::cout << ex << std::endl;
	exit(1);
    }
    return caster->GetOutput();
}

/* Explicit instantiations */
template plastimatch1_EXPORT UCharImageType::Pointer cast_uchar (FloatImageType::Pointer);
template plastimatch1_EXPORT ShortImageType::Pointer cast_short (UCharImageType::Pointer);
template plastimatch1_EXPORT ShortImageType::Pointer cast_short (ShortImageType::Pointer);
template plastimatch1_EXPORT ShortImageType::Pointer cast_short (UShortImageType::Pointer);
template plastimatch1_EXPORT ShortImageType::Pointer cast_short (ULongImageType::Pointer);
template plastimatch1_EXPORT ShortImageType::Pointer cast_short (FloatImageType::Pointer);
template plastimatch1_EXPORT FloatImageType::Pointer cast_float (FloatImageType::Pointer);
