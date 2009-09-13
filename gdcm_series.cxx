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
    gdcm_series_test_2 (dicom_dir);
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
gdcm_series_test_2 (char *dicom_dir)
{
    bool recursive = false;

    gdcm::SerieHelper2 *gdcm_shelper = new gdcm::SerieHelper2();

    gdcm_shelper->Clear ();
    gdcm_shelper->CreateDefaultUniqueSeriesIdentifier ();
    gdcm_shelper->SetUseSeriesDetails (true);
    gdcm_shelper->SetDirectory (dicom_dir, recursive);

    //gdcm_shelper->Print ();


    gdcm::FileList *file_list = gdcm_shelper->GetFirstSingleSerieUIDFileSet ();
    while (file_list) {
	if (file_list->size()) {	
	    gdcm_shelper->OrderFileList (file_list);

	    /* Choose one file, and print the id */
	    gdcm::File *file = (*file_list)[0];
	    std::string id = gdcm_shelper->
		    CreateUniqueSeriesIdentifier(file).c_str();
	    printf ("id = %s\n", id.c_str());

	    /* Iterate through files, and print the ipp */
	    print_series_ipp (file_list);

	}
	file_list = gdcm_shelper->GetNextSingleSerieUIDFileSet();
    }

    delete gdcm_shelper;
}

static void
gdcm_series_test_1 (char *dicom_dir)
{
    bool recursive = false;

    gdcm::SerieHelper *gdcm_shelper = new gdcm::SerieHelper();

    gdcm_shelper->Clear ();
    gdcm_shelper->SetUseSeriesDetails (false);
    gdcm_shelper->SetDirectory (dicom_dir, recursive);

    gdcm_shelper->Print ();

    gdcm::FileList *file_list = gdcm_shelper->GetFirstSingleSerieUIDFileSet ();
    while (file_list) {
	if (file_list->size()) {	
	    gdcm::File *file = (*file_list)[0]; //for example take the first one

	    std::string id = gdcm_shelper->
		    CreateUniqueSeriesIdentifier(file).c_str();

	    //printf ("id = %s\n", id.c_str());
	}
	file_list = gdcm_shelper->GetNextSingleSerieUIDFileSet();
    }

    delete gdcm_shelper;
}

/* Unfortunately, the itk layer is not rich enough to get information about 
    each image, such as its slice location */
static void
gdcm_series_itk_test (char *dicom_dir)
{
    typedef itk::GDCMSeriesFileNames NamesGeneratorType;
    NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
    //nameGenerator->SetUseSeriesDetails (true);
    //nameGenerator->SetUseSeriesDetails (false);
    /* GCS: The following is optional.  Do we need it? */
    // nameGenerator->AddSeriesRestriction("0008|0021" );
    nameGenerator->SetDirectory (dicom_dir);

    try {
	std::cout << std::endl << "The directory: " << std::endl;
	std::cout << std::endl << dicom_dir << std::endl << std::endl;
	std::cout << "Contains the following DICOM Series: ";
	std::cout << std::endl << std::endl;

	typedef std::vector< std::string > SeriesIdContainer;
	const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
	SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
	SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
	while (seriesItr != seriesEnd) {
	    std::cout << seriesItr->c_str() << std::endl;
	    seriesItr++;
	}

	/* Assumes series is first one found */
	std::string seriesIdentifier;
	seriesIdentifier = seriesUID.begin()->c_str();

	std::cout << std::endl << std::endl;
	std::cout << "Now reading series: " << std::endl << std::endl;
	std::cout << seriesIdentifier << std::endl;
	std::cout << std::endl << std::endl;

    } catch (itk::ExceptionObject &ex) {
	std::cout << ex << std::endl;
	print_and_exit ("Error loading dicom series.\n");
    }

}
