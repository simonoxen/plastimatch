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

	    /* Choose one file, and print the id */
	    gdcm::File *file = (*file_list)[0];
	    std::string id = this->gdcm_sh2->
		    CreateUniqueSeriesIdentifier(file).c_str();
	    printf ("id = %s\n", id.c_str());

	    /* Iterate through files, and print the ipp */
	    print_series_ipp (file_list);
	}
	file_list = this->gdcm_sh2->GetNextSingleSerieUIDFileSet();
    }

}
