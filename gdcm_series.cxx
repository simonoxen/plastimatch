/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "gdcmFile.h"
#include "gdcmFileHelper.h"
#include "gdcmGlobal.h"
#include "gdcmSeqEntry.h"
#include "gdcmSQItem.h"
#include "gdcmUtil.h"
#include "gdcm_series.h"
#include "gdcm_series_helper_2.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "print_and_exit.h"
#include "readcxt.h"

static void
gdcm_series_itk_test (char *dicom_dir);
static void
gdcm_series_test_1 (char *dicom_dir);
static void
gdcm_series_test_2 (char *dicom_dir);
static void
parse_directory (std::string const &dir, bool recursive);


plastimatch1_EXPORT
void
gdcm_series_test (char *dicom_dir)
{
    Gdcm_series gs;

    gs.load (dicom_dir);
}


static void
print_series_ipp (gdcm::FileList *file_list)
{
    // For all the files of a SingleSerieUID File set
    for (gdcm::FileList::iterator it =  file_list->begin();
	    it != file_list->end(); 
	    ++it)
    {
        double ipp[3];
	ipp[0] = (*it)->GetXOrigin();
	ipp[1] = (*it)->GetYOrigin();
	ipp[2] = (*it)->GetZOrigin();
	printf ("(%g,%g,%g)\t", ipp[0], ipp[1], ipp[2]);
	//printf ("Name = %s\n", (*it)->GetFileName().c_str());
    }
    printf ("\n");
}

static void
digest_file_list (gdcm::FileList *file_list, double origin[3], int dim[3], double spacing[3])
{
    int loop = 0;
    double prev_z;

    // For all the files of a SingleSerieUID File set
    for (gdcm::FileList::iterator it =  file_list->begin();
	    it != file_list->end(); 
	    ++it)
    {
	if (loop == 0) {
	    spacing[0] = (*it)->GetXSpacing ();
	    spacing[1] = (*it)->GetYSpacing ();
	    origin[0] = (*it)->GetXOrigin ();
	    origin[1] = (*it)->GetYOrigin ();
	    prev_z = origin[2] = (*it)->GetZOrigin ();
	    dim[0] = (*it)->GetXSize ();
	    dim[1] = (*it)->GetYSize ();
	    loop ++;
	} else if (loop == 1) {
	    double z = (*it)->GetZOrigin ();
	    if (z - prev_z > 1e-5) {
		spacing[2] = z - origin[2];
		loop ++;
	    } else {
		printf ("Warning: duplicate slice locations (%g)\n", z);
	    }
	    prev_z = z;
	} else {
	    double z = (*it)->GetZOrigin ();
	    if (z - prev_z > 1e-5) {
		//printf (">> %g %g %g\n", z, prev_z, spacing[2]);
		/* XiO rounds IPP to nearest .1 mm */
		if (fabs (z - prev_z - spacing[2]) > 0.11) {
		    print_and_exit ("Error: irregular slice thickness in dicom series\n");
		}
		loop ++;
	    } else {
		printf ("Warning: duplicate slice locations (%g)\n", z);
	    }
	    prev_z = z;
	}
    }
    dim[2] = loop;
}

Gdcm_series::Gdcm_series (void)
{
    this->gdcm_sh2 = 0;
}

Gdcm_series::~Gdcm_series (void)
{
    if (this->gdcm_sh2) {
	delete this->gdcm_sh2;
    }
}

void
Gdcm_series::load (char *dicom_dir)
{
    bool recursive = false;

    this->gdcm_sh2 = new gdcm::SerieHelper2();

    this->gdcm_sh2->Clear ();
    this->gdcm_sh2->CreateDefaultUniqueSeriesIdentifier ();
    this->gdcm_sh2->SetUseSeriesDetails (true);
    this->gdcm_sh2->SetDirectory (dicom_dir, recursive);

    //gdcm_shelper->Print ();


    gdcm::FileList *file_list = this->gdcm_sh2->GetFirstSingleSerieUIDFileSet ();
    while (file_list) {
	if (file_list->size()) {	
	    this->gdcm_sh2->OrderFileList (file_list);

#if defined (commentout)
	    /* Choose one file, and print the id */
	    gdcm::File *file = (*file_list)[0];
	    std::string id = this->gdcm_sh2->
		    CreateUniqueSeriesIdentifier(file).c_str();
	    printf ("id = %s\n", id.c_str());

	    /* Iterate through files, and print the ipp */
	    print_series_ipp (file_list);
#endif
	}
	file_list = this->gdcm_sh2->GetNextSingleSerieUIDFileSet();
    }

}

void
Gdcm_series::digest (void)
{
    int dim[3];
    double origin[3];
    double spacing[3];

    if (!this->gdcm_sh2) {
	return;
    }

    gdcm::FileList *file_list = this->gdcm_sh2->GetFirstSingleSerieUIDFileSet ();
    while (file_list) {
	if (file_list->size()) {	
	    this->gdcm_sh2->OrderFileList (file_list);

	    /* Get the USI */
	    gdcm::File *file = (*file_list)[0];
	    std::string id = this->gdcm_sh2->
		    CreateUniqueSeriesIdentifier(file).c_str();

	    /* Digest the USI */
	    digest_file_list (file_list, origin, dim, spacing);
	    printf ("---- %s\n", id.c_str());
	    printf ("DIM = %d %d %d\n", dim[0], dim[1], dim[2]);
	    printf ("OFF = %g %g %g\n", origin[0], origin[1], origin[2]);
	    printf ("SPA = %g %g %g\n", spacing[0], spacing[1], spacing[2]);


	}
	file_list = this->gdcm_sh2->GetNextSingleSerieUIDFileSet();
    }
}
