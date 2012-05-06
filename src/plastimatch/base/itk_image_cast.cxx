/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#if (defined(_WIN32) || defined(WIN32))
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#endif

#include "plmbase.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkOrientImageFilter.h"

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
CharImageType::Pointer
cast_char (T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ClampCastImageFilter <
	ImageType, CharImageType > ClampCastFilterType;

    typename ClampCastFilterType::Pointer caster = ClampCastFilterType::New ();
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
UCharImageType::Pointer
cast_uchar (T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ClampCastImageFilter <
	ImageType, UCharImageType > ClampCastFilterType;

    typename ClampCastFilterType::Pointer caster = ClampCastFilterType::New();
    caster->SetInput(image);
    try {
	caster->Update();
    }
    catch (itk::ExceptionObject & ex) {
	printf ("ITK exception in ClampCastFilter.\n");
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
    typedef itk::ClampCastImageFilter <
	ImageType, UShortImageType > ClampCastFilterType;

    typename ClampCastFilterType::Pointer caster = ClampCastFilterType::New();
    caster->SetInput(image);
    try {
	caster->Update();
    }
    catch (itk::ExceptionObject & ex) {
	printf ("ITK exception in ClampCastFilter.\n");
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
    typedef itk::ClampCastImageFilter <
	ImageType, ShortImageType > ClampCastFilterType;

    typename ClampCastFilterType::Pointer caster = ClampCastFilterType::New();
    caster->SetInput(image);
    try {
	caster->Update();
    }
    catch (itk::ExceptionObject & ex) {
	printf ("ITK exception in ClampCastFilter.\n");
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
    typedef itk::ClampCastImageFilter <
	ImageType, Int32ImageType > ClampCastFilterType;

    typename ClampCastFilterType::Pointer caster = ClampCastFilterType::New();
    caster->SetInput(image);
    try {
	caster->Update();
    }
    catch (itk::ExceptionObject & ex) {
	printf ("ITK exception in ClampCastFilter.\n");
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
    typedef itk::ClampCastImageFilter <
	ImageType, UInt32ImageType > ClampCastFilterType;

    typename ClampCastFilterType::Pointer caster = ClampCastFilterType::New();
    caster->SetInput(image);
    try {
	caster->Update();
    }
    catch (itk::ExceptionObject & ex) {
	printf ("ITK exception in ClampCastFilter.\n");
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
    typedef itk::ClampCastImageFilter <
	ImageType, FloatImageType > ClampCastFilterType;

    typename ClampCastFilterType::Pointer caster = ClampCastFilterType::New();
    caster->SetInput(image);
    try {
	caster->Update();
    }
    catch (itk::ExceptionObject & ex) {
	printf ("ITK exception in ClampCastFilter.\n");
	std::cout << ex << std::endl;
	exit(1);
    }
    return caster->GetOutput();
}

template<class T> 
DoubleImageType::Pointer
cast_double (T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ClampCastImageFilter <
	ImageType, DoubleImageType > ClampCastFilterType;

    typename ClampCastFilterType::Pointer caster = ClampCastFilterType::New();
    caster->SetInput(image);
    try {
	caster->Update();
    }
    catch (itk::ExceptionObject & ex) {
	printf ("ITK exception in ClampCastFilter.\n");
	std::cout << ex << std::endl;
	exit(1);
    }
    return caster->GetOutput();
}

/* Explicit instantiations */
template API CharImageType::Pointer cast_char (ShortImageType::Pointer);
template API CharImageType::Pointer cast_char (FloatImageType::Pointer);
template API UCharImageType::Pointer cast_uchar (ShortImageType::Pointer);
template API UCharImageType::Pointer cast_uchar (FloatImageType::Pointer);
template API ShortImageType::Pointer cast_short (UCharImageType::Pointer);
template API ShortImageType::Pointer cast_short (ShortImageType::Pointer);
template API ShortImageType::Pointer cast_short (UShortImageType::Pointer);
template API ShortImageType::Pointer cast_short (Int32ImageType::Pointer);
template API ShortImageType::Pointer cast_short (UInt32ImageType::Pointer);
template API ShortImageType::Pointer cast_short (FloatImageType::Pointer);
template API UShortImageType::Pointer cast_ushort (ShortImageType::Pointer);
template API UShortImageType::Pointer cast_ushort (FloatImageType::Pointer);
template API Int32ImageType::Pointer cast_int32 (ShortImageType::Pointer);
template API Int32ImageType::Pointer cast_int32 (FloatImageType::Pointer);
template API UInt32ImageType::Pointer cast_uint32 (UCharImageType::Pointer);
template API UInt32ImageType::Pointer cast_uint32 (ShortImageType::Pointer);
template API UInt32ImageType::Pointer cast_uint32 (FloatImageType::Pointer);
template API FloatImageType::Pointer cast_float (UCharImageType::Pointer);
template API FloatImageType::Pointer cast_float (ShortImageType::Pointer);
template API FloatImageType::Pointer cast_float (UShortImageType::Pointer);
template API FloatImageType::Pointer cast_float (UInt32ImageType::Pointer);
template API FloatImageType::Pointer cast_float (FloatImageType::Pointer);
template API DoubleImageType::Pointer cast_double (UCharImageType::Pointer);
template API DoubleImageType::Pointer cast_double (ShortImageType::Pointer);
template API DoubleImageType::Pointer cast_double (UInt32ImageType::Pointer);
template API DoubleImageType::Pointer cast_double (FloatImageType::Pointer);
