/** @file compute_mean_mha_main
 *  @brief Generate a mean image from a set of registered images 
 */
#include <time.h>
#include <stdlib.h>
#include <string.h>
  
#include "plm_config.h"
#include "itkImage.h"
#include "itkAddImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkNaryAddImageFilter.h"
#include "itkDivideByConstantImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"

#include "itk_image.h"
#include "getopt.h"


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

void parse_filelist(const char* fName, char***fNameList, int *nFiles)
{
	int i;
	int numFiles = 0;

	FILE* fp = fopen(fName, "r");
	// file pointer is NULL
	if (!fp)
	{
		fprintf(stderr, "FILE %s open failed!!\n", fName);
		exit(-1);
	}  
	
	fscanf(fp, "%d", &numFiles);
	fprintf(stderr, "%d \n", numFiles); 
	(*nFiles) = numFiles; 	
	(*fNameList) = (char**) malloc ( sizeof(char*) * numFiles);			
	
	for (i = 0; i < numFiles; i ++)
		(*fNameList)[i] = (char *) malloc ( sizeof(char) * 256);	
	
	for (i = 0; i < numFiles; i ++)	
		fscanf(fp, "%s", (*fNameList)[i]); 

	fclose(fp);
} 

void print_filelist(char** fNameList, int nFiles)
{
	int i;

	for (i = 0; i < nFiles; i ++)
		fprintf(stdout, "%s \n", fNameList[i]);
}

void compute_average(char **inFileList, int nFiles, char *resFile)
{
	typedef itk::NaryAddImageFilter< ShortImageType, ShortImageType > 
		AddFilterType;
	typedef itk::DivideByConstantImageFilter< ShortImageType, int, ShortImageType >
		DivFilterType;
	typedef itk::ImageFileReader< ShortImageType > ReaderType;
	typedef itk::ImageFileWriter< ShortImageType > WriterType;

	ReaderType::Pointer reader = ReaderType::New();
	AddFilterType::Pointer addition = AddFilterType::New();
	DivFilterType::Pointer division = DivFilterType::New();

	division->SetConstant(nFiles);
	
	//add all the input images
	for (int i = 0; i < nFiles; i ++)
	{
		reader->SetFileName(inFileList[i]);
		reader->Update();
		// do division first
		division->SetInput(reader->GetOutput());
		division->Update();
		addition->SetInput(i, division->GetOutput());
	}
	addition->Update();

	//write the output file	
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(resFile);
	writer->SetInput(addition->GetOutput());
	std::cout << "Write file ..." << std::endl << std::endl;
	writer->Update();	
	std::cout << "File " << resFile << " created" << std::endl;
}

int main (int argc, char *argv[])
{
	//list of fnames -- to compute average
	char **fNameList; 

	//number of image files
	int nFiles;

	if (argc < 3)
		print_usage();
	else			        		
	{
		//parse the file list
		parse_filelist(argv[1], &fNameList, &nFiles);		

		//print the input file list
		fprintf(stdout, "Reading in the list of files ...\n");
		print_filelist(fNameList, nFiles);
		fprintf(stdout, "Reading in the list of files ... DONE!! \n");
		//compute the average image
		compute_average(fNameList, nFiles, argv[2]);		
	}	
}
