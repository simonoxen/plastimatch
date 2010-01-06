/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string>
#include <itksys/SystemTools.hxx>
#include <itkImageIOBase.h>

#include "file_type.h"
#include "itk_image.h"
#include "gdcm_rtss.h"

Plm_file_type
deduce_file_type (char* path)
{
    std::string ext;
    
    if (itksys::SystemTools::FileIsDirectory (path)) {
	/* GCS TODO:  Distinguish between xio, dicom, rtog directories */
	return PLM_FILE_TYPE_DICOM_DIR;
    }

    ext = itksys::SystemTools::GetFilenameLastExtension (std::string (path));

    if (!itksys::SystemTools::Strucmp (ext.c_str(), ".txt")) {
	/* Probe for pointset */
	int rc;
	const int MAX_LINE = 2048;
	char line[MAX_LINE];
	float f[4];
	FILE* fp = fopen (path, "rb");
	if (!fp) return PLM_FILE_TYPE_NO_FILE;

	fgets (line, MAX_LINE, fp);
	fclose (fp);

	rc = sscanf (line, "%g %g %g %g", &f[0], &f[1], &f[2], &f[3]);
	if (rc == 3) {
	    return PLM_FILE_TYPE_POINTSET;
	}

	/* Not sure, assume image */
	return PLM_FILE_TYPE_IMG;
    }

    if (!itksys::SystemTools::Strucmp (ext.c_str(), ".cxt")) {
	return PLM_FILE_TYPE_CXT;
    }

    if (!itksys::SystemTools::Strucmp (ext.c_str(), ".dij")) {
	return PLM_FILE_TYPE_DIJ;
    }

    if (!itksys::SystemTools::Strucmp (ext.c_str(), ".pfm")) {
	return PLM_FILE_TYPE_PROJ_IMG;
    }
    if (!itksys::SystemTools::Strucmp (ext.c_str(), ".hnd")) {
	return PLM_FILE_TYPE_PROJ_IMG;
    }


    /* Maybe vector field? */
    itk::ImageIOBase::IOPixelType pixelType;
    itk::ImageIOBase::IOComponentType componentType;

    printf ("Testing vf type.\n");
    itk__GetImageType (std::string (path), pixelType, componentType);
    if (pixelType == itk::ImageIOBase::VECTOR) {
	return PLM_FILE_TYPE_VF;
    }

    /* Maybe dicom rtss? */
    printf ("Testing rtss.\n");
    if (gdcm_rtss_probe (path)) {
	return PLM_FILE_TYPE_DICOM_RTSS;
    }

    return PLM_FILE_TYPE_IMG;
}

char*
file_type_string (Plm_file_type file_type)
{
    switch (file_type) {
    case PLM_FILE_TYPE_NO_FILE:
	return "No such file";
	break;
    case PLM_FILE_TYPE_UNKNOWN:
	return "Unknown";
	break;
    case PLM_FILE_TYPE_IMG:
	return "Image";
	break;
    case PLM_FILE_TYPE_DIJ:
	return "Dij matrix";
	break;
    case PLM_FILE_TYPE_POINTSET:
	return "Pointset";
	break;
    case PLM_FILE_TYPE_CXT:
	return "Cxt file";
	break;
    case PLM_FILE_TYPE_DICOM_DIR:
	return "Dicom directory";
	break;
    case PLM_FILE_TYPE_XIO_DIR:
	return "XiO directory";
	break;
    case PLM_FILE_TYPE_RTOG_DIR:
	return "RTOG directory";
	break;
    case PLM_FILE_TYPE_PROJ_IMG:
	return "Projection image";
	break;
    default:
	return "Unknown/default";
	break;
    }
}
