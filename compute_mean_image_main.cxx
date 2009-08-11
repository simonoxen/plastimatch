/*! \file compute_mean_image_main
 *  \brief Generate a mean image from a set of images 
 *
 *  This funciton can compute the mean image from a
 *  set of CT images, either of format .mha, or .dcm
 *  or .nrrd
 *
 *  Author: Rui Li, Greg Sharp
 *  
 *  Date Created: June 26, 2009
 *  Last Modified: July 23, 2009
 */
#include <string.h>

#include <iostream>
#include <fstream>

#if (defined(_WIN32) || defined(WIN32))
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#endif

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

#include "plm_path.h"
#include "itk_image.h"
#include "itk_dicom.h"
#include "getopt.h"

/* local functions used in itk_image.cxx -- 
  package to a separate plm_utils.h and plm_utils.cxx might be a better
  way for this type of utility functions
*/

// check if the input string is a directory
static int is_directory (char *dir)
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

// print out the image names/directoies stored 
// in imageList
void print_image_list(char** imageList, int nImages)
{
	int i;

	for (i = 0; i < nImages; i ++)
		fprintf(stdout, "%s \n", imageList[i]);
}

// parse the list of file/directory names, store them in imageList
// and the number of images in nImages
void parse_image_list(const char *fName, char ***imageList, int *nImages)
{
    FILE* fp = fopen(fName, "r");
    char curLine[_MAX_PATH];
    int nLines = 0;
    
    // file pointer is NULL
    if (!fp)
    {
	    fprintf(stderr, "FILE %s open failed!!\n", fName);
	    exit(-1);
    }  

    // initialize the imageList
    *imageList = NULL;
    while (fgets (curLine, _MAX_PATH, fp)) 
    {
	    (*imageList) = (char**) realloc ((*imageList), (nLines+1) * sizeof(char**));
	    (*imageList)[nLines] = (char*) malloc (strlen(curLine)+1);
	    strcpy ((*imageList)[nLines], curLine);
        // remove the '\n' character if there is one at the end of the line
        if (curLine[strlen(curLine) - 1] == '\n')
            (*imageList)[nLines][strlen((*imageList)[nLines])-1] = 0;
	    nLines++;
    }
    *nImages = nLines;
    fclose (fp);
}

void compute_average(char **imageList, int nImages, char *outFile)
{
    // filters we need to compute mean image
    // we use FloatImageType to avoid pixel value overflow during the addition
    typedef itk::AddImageFilter< FloatImageType, FloatImageType, FloatImageType > 
        AddFilterType;
    typedef itk::DivideByConstantImageFilter< FloatImageType, int, FloatImageType >
        DivFilterType;

    // the original type of the image
    PlmImageType origImageType;

    FloatImageType::Pointer tmp;
    FloatImageType::Pointer sumImg;

	AddFilterType::Pointer addition = AddFilterType::New();
	DivFilterType::Pointer division = DivFilterType::New();

    if (nImages <= 1) 
    {
        std::cout << "number of images is less than or equal to 1" << std::endl;
        return;
    }

	sumImg = load_float (imageList[0], &origImageType);

	//add all the input images
	for (int i = 1; i < nImages; i ++)
	{
	    tmp = load_float (imageList[i], &origImageType);
	    addition->SetInput1 (sumImg);
	    addition->SetInput2 (tmp);
	    addition->Update();
	    sumImg = addition->GetOutput ();
	}

    // divide by the total number of input images
    division->SetConstant(nImages);
	division->SetInput (sumImg);
	division->Update();
    // store the mean image in tmp first before write out
    tmp = division->GetOutput();

    // write the computed mean image
    if (is_directory(outFile)) 
    {
        std::cout << "output dicom to " << outFile << std::endl;
        // Dicom
        save_short_dicom (tmp, outFile);
    }
    else
    {
        std::cout << "output to " << outFile << std::endl;
        save_short(tmp, outFile);
    }

    // free allocated memeory 
    for (int i = 0; i < nImages; i ++)
         free(imageList[i]);
    free(imageList);
}

int main (int argc, char *argv[])
{
    // list of file names or list of dicom directories
    // for mean image computation
    // E.g. imageList can be a list of file names for mhd
    //      files, nrrnd files or a list of directories 
    //      for DICOM format
    char **imageList;
    int nImages;

    // check for input arguments
    if (argc < 3)
    {
        printf("Usage: compute_mean_image_main [list file] [result file] \n");
        exit(1);    
    }
    else { 
        // parse the list
        parse_image_list(argv[1], &imageList, &nImages);

        // print the input image list
        print_image_list(imageList, nImages);

        // compute and write out the average image
        compute_average(imageList, nImages, argv[2]);		
    }
}

