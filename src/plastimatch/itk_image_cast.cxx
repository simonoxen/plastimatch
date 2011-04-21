/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#if (defined(_WIN32) || defined(WIN32))
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#endif
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkOrientImageFilter.h"
#include "itk_image.h"
#include "logfile.h"
#include "print_and_exit.h"

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
UShortImageType::Pointer
cast_ushort (T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::CastImageFilter <
	ImageType, UShortImageType > CastFilterType;

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
Int32ImageType::Pointer
cast_int32 (T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::CastImageFilter <
	ImageType, Int32ImageType > CastFilterType;

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
UInt32ImageType::Pointer
cast_uint32 (T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::CastImageFilter <
	ImageType, UInt32ImageType > CastFilterType;

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
template plastimatch1_EXPORT ShortImageType::Pointer cast_short (Int32ImageType::Pointer);
template plastimatch1_EXPORT ShortImageType::Pointer cast_short (UInt32ImageType::Pointer);
template plastimatch1_EXPORT ShortImageType::Pointer cast_short (FloatImageType::Pointer);
template plastimatch1_EXPORT UShortImageType::Pointer cast_ushort (FloatImageType::Pointer);
template plastimatch1_EXPORT Int32ImageType::Pointer cast_int32 (FloatImageType::Pointer);
template plastimatch1_EXPORT UInt32ImageType::Pointer cast_uint32 (UCharImageType::Pointer);
template plastimatch1_EXPORT UInt32ImageType::Pointer cast_uint32 (ShortImageType::Pointer);
template plastimatch1_EXPORT UInt32ImageType::Pointer cast_uint32 (FloatImageType::Pointer);
template plastimatch1_EXPORT FloatImageType::Pointer cast_float (UCharImageType::Pointer);
template plastimatch1_EXPORT FloatImageType::Pointer cast_float (ShortImageType::Pointer);
template plastimatch1_EXPORT FloatImageType::Pointer cast_float (UInt32ImageType::Pointer);
template plastimatch1_EXPORT FloatImageType::Pointer cast_float (FloatImageType::Pointer);
