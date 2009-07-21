/** @file compute_mean_mha_main
 *  @brief Generate a mean image from a set of registered images 
 */
#include <string.h>
#include <direct.h>

#include <iostream>
#include <fstream>

#include "plm_config.h"
#include "itkImage.h"
#include "itkImageRegion.h"
#include "itkAddImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkNaryAddImageFilter.h"
#include "itkDivideByConstantImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkGDCMImageIO.h" 
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"

#include "itk_image.h"
#include "itk_dicom.h"
#include "getopt.h"

void print_filelist(char** fNameList, int nFiles);

static int
_is_directory (char *dir)
{
#if (defined(_WIN32) || defined(WIN32))
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



void print_image(ShortImageType::Pointer image, char *fname)
{
        typedef itk::ImageRegionConstIterator< ShortImageType > ConstIteratorType;
        ConstIteratorType itg(image, image->GetRequestedRegion());

        int count = 0;

        std::ofstream out_file(fname, std::ios::out|std::ios::app|std::ios::ate);
        if (out_file.bad(  ))
                return; /* Where do we log an error if there is no log */
   
        for (itg.GoToBegin(); !itg.IsAtEnd(); ++itg)
        {
                out_file << itg.Get() << "\t";
                count ++;
                if (count%512 == 0)
                        out_file << std::endl;
        }

}

bool getFileExtension(const char *filename)
{
        int len = strlen(filename);
	char *ext = (char *)malloc(sizeof(char)*3);
        filename += len-3;
        strcpy(ext,filename);
	bool isDicom = false;

        if (strcmp(ext, "dcm") == 0 || strcmp(ext, "DCM") == 0)
        {
                        isDicom = true;                        
                        std::cout << "isDicom set to true!" << std::endl;
        }

        return isDicom;
}

void print_usage (void)
{
	printf("Usage: compute_mean_mha [list file] [result file] \n");
	exit(1);
}


void show_stats(ShortImageType::Pointer image)
{	
	ShortImageType::RegionType region = image->GetLargestPossibleRegion();
	const ShortImageType::IndexType& st = region.GetIndex();
	const ShortImageType::SizeType& sz = image->GetLargestPossibleRegion().GetSize();
	const ShortImageType::PointType& og =  image->GetOrigin();
	const ShortImageType::SpacingType& sp = image->GetSpacing();

	printf ("Origin = %g %g %g\n", og[0], og[1], og[2]);
	printf ("Spacing = %g %g %g\n", sp[0], sp[1], sp[2]);
	std::cout << "Start = " << st[0] << " " << st[1] << " " << st[2] << std::endl;
	std::cout << "Size = " << sz[0] << " " << sz[1] << " " << sz[2] << std::endl;
}

void parse_filelist(const char* fName, char***fDirList, int *nFiles)
{
    int i = 0;
    int numFiles = 0;
    char buf[2048];

    FILE* fp = fopen(fName, "r");
    // file pointer is NULL
    if (!fp)
    {
	    fprintf(stderr, "FILE %s open failed!!\n", fName);
	    exit(-1);
    }  
    (*fDirList) = 0;
    while (fgets (buf, 2048, fp)) {
	(*fDirList) = (char**) realloc ((*fDirList), (i+1) * sizeof(char**));
	(*fDirList)[i] = (char*) malloc (strlen(buf)+1);
	strcpy ((*fDirList)[i], buf);
	(*fDirList)[i][strlen((*fDirList)[i])-1] = 0;
	i++;
    }
    *nFiles = i;
    fclose (fp);
} 

void print_filelist(char** fNameList, int nFiles)
{
	int i;

	for (i = 0; i < nFiles; i ++)
		fprintf(stdout, "%s \n", fNameList[i]);
}

void compute_average(char **inFileList, int nFiles, char *resFile, bool isDicom)
{
	//typedef itk::NaryAddImageFilter< ShortImageType, ShortImageType > 
	//	AddFilterType;

        typedef itk::AddImageFilter< FloatImageType, FloatImageType, FloatImageType > 
		AddFilterType;
        typedef itk::DivideByConstantImageFilter< FloatImageType, int, FloatImageType >
		DivFilterType;
	typedef itk::ImageFileReader< FloatImageType > ReaderType;
	typedef itk::ImageFileWriter< FloatImageType > WriterType;
        typedef itk::GDCMImageIO ImageIOType; 
        typedef itk::MetaDataDictionary DictionaryType;
        FloatImageType::Pointer tmp;
        FloatImageType::Pointer sumImg;
    PlmImageType original_type;
#if defined (commentout)
        ImageIOType::Pointer gdcmImageIO = ImageIOType::New(); 

        ReaderType::Pointer reader = ReaderType::New();
#endif

	AddFilterType::Pointer addition = AddFilterType::New();
	DivFilterType::Pointer division = DivFilterType::New();

    if (nFiles <= 0) return;

    printf ("%s -> %d\n", inFileList[0], _is_directory(inFileList[0]));
	sumImg = load_float (inFileList[0], &original_type);
	//add all the input images
	for (int i = 1; i < nFiles; i ++)
	{
	    tmp = load_float (inFileList[i], &original_type);
	    addition->SetInput1 (sumImg);
	    addition->SetInput2 (tmp);
	    addition->Update();
	    sumImg = addition->GetOutput ();
	}

	division->SetConstant(nFiles);
	division->SetInput (sumImg);
	division->Update();
        tmp = division->GetOutput ();
	save_short_dicom (tmp, "C:/tmp/junk");

        //free memory for file name list
        for (int i = 0; i < nFiles; i ++)
                free(inFileList[i]);
        free(inFileList);
}

int main (int argc, char *argv[])
{
	// list of fnames -- to compute average
	char **fDirList; 
        char *buffer;
	// number of image files
	int nFiles;

        // flag to indicate whether this is a dicom file
        // default value is set to false
        bool isDicom = false;

	if (argc < 3)
		print_usage();
	else			        		
	{
                buffer = _getcwd( NULL, 0 );

		//parse the file list
		parse_filelist(argv[1], &fDirList, &nFiles);		

		//print the input file list
		print_filelist(fDirList, nFiles);

                //check whether this is a dicom file
                //isDicom = getFileExtension(fNameList[0]);

                //compute the average image
		compute_average(fDirList, nFiles, argv[2], isDicom);		
	}	
}
