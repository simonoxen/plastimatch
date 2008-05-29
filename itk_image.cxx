/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include <stdlib.h>
#include <string.h>
#if defined (WIN32)
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#endif
#include "config.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDICOMImageIO2.h"
#include "itkImageSeriesReader.h"
#include "itkDICOMSeriesFileNames.h"
#include "itkCastImageFilter.h"
#include "itk_image.h"

/* =======================================================================*
    Definitions
 * =======================================================================*/
typedef itk::ImageSeriesReader < ShortImageType > DicomShortReaderType;
typedef itk::ImageSeriesReader < UShortImageType > DicomUShortReaderType;
typedef itk::ImageSeriesReader < FloatImageType > DicomFloatReaderType;

typedef itk::ImageFileReader < ShortImageType > MhaShortReaderType;
typedef itk::ImageFileReader < UShortImageType > MhaUShortReaderType;
typedef itk::ImageFileReader < UCharImageType > MhaUCharReaderType;
typedef itk::ImageFileReader < FloatImageType > MhaFloatReaderType;


/* =======================================================================*
    Functions
 * =======================================================================*/
int
is_directory (char *dir)
{
#if defined (WIN32)
    char pwd[_MAX_PATH];
    if (!_getcwd (pwd, _MAX_PATH)) {
        return 0;
    }
    if (_chdir (dir) == -1) {
        return 0;
    }
    _chdir (pwd);
#else /* UNIX */
    DIR *dp;
    if ((dp = opendir (dir)) == NULL) {
        return 0;
    }
    closedir (dp);
#endif
    return 1;
}

template<class RdrT>
void
load_mha_rdr(RdrT reader, char *fn)
{
    reader->SetFileName(fn);
    try {
	reader->Update();
    }
    catch(itk::ExceptionObject & ex) {
	printf ("Exception reading mha file: %s!\n",fn);
	std::cout << ex << std::endl;
	getchar();
	exit(1);
    }
}

template<class T>
void
load_dicom_dir_rdr(T rdr, char *dicom_dir)
{
    itk::DICOMImageIO2::Pointer dicomIO = itk::DICOMImageIO2::New();

    // Get the DICOM filenames from the directory
    itk::DICOMSeriesFileNames::Pointer nameGenerator =
	    itk::DICOMSeriesFileNames::New();
    nameGenerator->SetDirectory(dicom_dir);

    typedef std::vector < std::string > seriesIdContainer;
    const seriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
    seriesIdContainer::const_iterator seriesItr = seriesUID.begin();
    seriesIdContainer::const_iterator seriesEnd = seriesUID.end();
    std::cout << std::endl << "The directory: " << std::endl;
    std::cout << std::endl << dicom_dir << std::endl << std::endl;
    std::cout << "Contains the following DICOM Series: ";
    std::cout << std::endl << std::endl;

    while (seriesItr != seriesEnd) {
	std::cout << seriesItr->c_str() << std::endl;
	seriesItr++;
    }

    std::cout << std::endl << std::endl;
    std::cout << "Now reading series: " << std::endl << std::endl;

    typedef std::vector < std::string > fileNamesContainer;
    fileNamesContainer fileNames;

    std::cout << seriesUID.begin()->c_str() << std::endl;
    fileNames = nameGenerator->GetFileNames();

    rdr->SetFileNames(fileNames);
    rdr->SetImageIO(dicomIO);

    try {
	rdr->Update();
    }
    catch(itk::ExceptionObject & ex) {
	printf ("Exception reading dicom directory: %s!\n",dicom_dir);
	std::cout << ex << std::endl;
	exit(1);
    }
}

ShortImageType::Pointer
load_dicom_short (char *dicom_dir)
{
    DicomShortReaderType::Pointer fixed_input_rdr
		= DicomShortReaderType::New();
    load_dicom_dir_rdr(fixed_input_rdr, dicom_dir);
    fixed_input_rdr->Update();
    return fixed_input_rdr->GetOutput();
}

UShortImageType::Pointer
load_dicom_ushort (char *dicom_dir)
{
    DicomUShortReaderType::Pointer fixed_input_rdr
		= DicomUShortReaderType::New();
    load_dicom_dir_rdr(fixed_input_rdr, dicom_dir);
    fixed_input_rdr->Update();
    return fixed_input_rdr->GetOutput();
}

int
get_mha_type (char* mha_fname)
{
    char buf[1024];
    FILE* fp = fopen (mha_fname, "r");
    if (!fp) {
	printf ("Could not open mha file for read\n");
	exit (-1);
    }
    while (fgets(buf,1024,fp)) {
	if (!strcmp(buf, "ElementType = MET_SHORT\n")) {
	    return TYPE_SHORT;
	} else if (!strcmp(buf, "ElementType = MET_USHORT\n")) {
	    return TYPE_USHORT;
	} else if (!strcmp(buf, "ElementType = MET_UCHAR\n")) {
	    return TYPE_UCHAR;
	} else if (!strcmp(buf, "ElementType = MET_FLOAT\n")) {
	    return TYPE_FLOAT;
	} else if (!strncmp(buf,"ElementType",sizeof("ElementType"))) {
	    printf ("No ElementType in mha file\n");
	    exit (-1);
	}
    }
    printf ("No ElementType in mha file\n");
    exit (-1);
    return 0;  /* Get rid of warning */
}

template<class T>
FloatImageType::Pointer
load_mha_float_2 (T rdr, char* mha_fname)
{
    typedef typename T::ObjectType::OutputImagePixelType PixelType;
    typedef typename itk::Image < PixelType, 3 > InputImageType;
    typedef typename itk::CastImageFilter < 
		InputImageType, FloatImageType > CastFilterType;

    load_mha_rdr (rdr, mha_fname);
    typename InputImageType::Pointer input_image = rdr->GetOutput();

    /* Convert images to float */
    typename CastFilterType::Pointer caster = CastFilterType::New();
    caster->SetInput (input_image);

    typename FloatImageType::Pointer image = caster->GetOutput();
    image->Update();
    return image;
}

FloatImageType::Pointer
load_mha_float (char* mha_fname)
{
    int file_type = get_mha_type (mha_fname);
    if (file_type == TYPE_SHORT) {
	MhaShortReaderType::Pointer rdr = MhaShortReaderType::New();
	return load_mha_float_2(rdr, mha_fname);
    } else if (file_type == TYPE_USHORT) {
	MhaUShortReaderType::Pointer rdr = MhaUShortReaderType::New();
	return load_mha_float_2(rdr, mha_fname);
    } else if (file_type == TYPE_FLOAT) {
	MhaFloatReaderType::Pointer rdr = MhaFloatReaderType::New();
	return load_mha_float_2(rdr, mha_fname);
    } else {
	printf ("Load conversion failure (unsupported).\n");
	exit (-1);
    }
}

template<class T>
ShortImageType::Pointer
load_mha_short_2 (T rdr, char* mha_fname)
{
    typedef typename T::ObjectType::OutputImagePixelType PixelType;
    typedef typename itk::Image < PixelType, 3 > InputImageType;
    typedef typename itk::CastImageFilter < 
		InputImageType, ShortImageType > CastFilterType;

    load_mha_rdr (rdr, mha_fname);
    typename InputImageType::Pointer input_image = rdr->GetOutput();

    /* Convert images to float */
    typename CastFilterType::Pointer caster = CastFilterType::New();
    caster->SetInput (input_image);

    typename ShortImageType::Pointer image = caster->GetOutput();
    image->Update();
    return image;
}

ShortImageType::Pointer
load_mha_short (char* mha_fname)
{
    int file_type = get_mha_type (mha_fname);
    if (file_type == 1) {
	MhaShortReaderType::Pointer rdr = MhaShortReaderType::New();
	return load_mha_short_2(rdr, mha_fname);
    } else {
	MhaUShortReaderType::Pointer rdr = MhaUShortReaderType::New();
	return load_mha_short_2(rdr, mha_fname);
    }
}

ShortImageType::Pointer
load_short (char* fname)
{
    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	return load_dicom_short (fname);
    } else {
	return load_mha_short (fname);
    }
}

template<class T>
UCharImageType::Pointer
load_mha_uchar_2 (T rdr, char* mha_fname)
{
    typedef typename T::ObjectType::OutputImagePixelType PixelType;
    typedef typename itk::Image < PixelType, 3 > InputImageType;
    typedef typename itk::CastImageFilter < 
		InputImageType, UCharImageType > CastFilterType;

    load_mha_rdr (rdr, mha_fname);
    typename InputImageType::Pointer input_image = rdr->GetOutput();

    /* Convert images to uchar */
    typename CastFilterType::Pointer caster = CastFilterType::New();
    caster->SetInput (input_image);

    UCharImageType::Pointer image = caster->GetOutput();
    image->Update();
    return image;
}

UCharImageType::Pointer
load_mha_uchar (char* mha_fname)
{
    int file_type = get_mha_type (mha_fname);
    if (file_type == 3) {
	MhaUCharReaderType::Pointer rdr = MhaUCharReaderType::New();
	return load_mha_uchar_2(rdr, mha_fname);
    } else {
	/* This won't work */
	MhaUCharReaderType::Pointer rdr = MhaUCharReaderType::New();
	return load_mha_uchar_2(rdr, mha_fname);
    }
}

UCharImageType::Pointer
load_uchar (char* fname)
{
    /* Dicom not yet supported */
    return load_mha_uchar (fname);
}

FloatImageType::Pointer
load_float (char* fname)
{
    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	/* Not yet implemented */
	printf ("Sorry, cannot load dicom as float\n");
	exit (-1);
    } else {
	return load_mha_float (fname);
    }
}

DeformationFieldType::Pointer
load_float_field (char* fname)
{
    typedef itk::ImageFileReader< DeformationFieldType >  FieldReaderType;

    FieldReaderType::Pointer fieldReader = FieldReaderType::New();
    fieldReader->SetFileName (fname);

    try 
    {
	fieldReader->Update();
    }
    catch (itk::ExceptionObject& excp) 
    {
	std::cerr << "Exception thrown " << std::endl;
	std::cerr << excp << std::endl;
	return 0;
    }
    DeformationFieldType::Pointer deform_field = fieldReader->GetOutput();
    return deform_field;
}

template<class T> 
void
save_image (T image, char* fname)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ImageFileWriter< ImageType >  WriterType;

    printf ("Trying to write image to %s\n", fname);

    typename WriterType::Pointer writer = WriterType::New();
    writer->SetInput (image);
    writer->SetFileName (fname);
    try {
	writer->Update();
    }
    catch (itk::ExceptionObject& excp) {
	std::cerr << "Exception thrown " << std::endl;
	std::cerr << excp << std::endl;
    }
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
    caster->Update();
    return caster->GetOutput();
}

template<class T> 
void
save_short (T image, char* fname)
{
    ShortImageType::Pointer short_img = cast_short(image);
    save_image (short_img, fname);
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
    caster->Update();
    return caster->GetOutput();
}

template<class T> 
void
save_float (T image, char* fname)
{
    FloatImageType::Pointer float_img = cast_float(image);
    save_image (float_img, fname);
}


/* Explicit instantiations */
template void load_mha_rdr (MhaUCharReaderType::Pointer reader, char *fn);
template void load_dicom_dir_rdr(DicomShortReaderType::Pointer rdr, char *dicom_dir);
template void load_dicom_dir_rdr(DicomUShortReaderType::Pointer rdr, char *dicom_dir);
template void load_dicom_dir_rdr(DicomFloatReaderType::Pointer rdr, char *dicom_dir);
template void save_image(FloatImageType::Pointer, char*);
template void save_image(ShortImageType::Pointer, char*);
template void save_image(UCharImageType::Pointer, char*);
template void save_image(DeformationFieldType::Pointer, char*);
template void save_short(FloatImageType::Pointer, char*);
template void save_float(FloatImageType::Pointer, char*);
