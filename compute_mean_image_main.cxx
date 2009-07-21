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
#include "getopt.h"

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
		//fgets((*fNameList)[i], 256, fp);

	fclose(fp);
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

        typedef itk::AddImageFilter< ShortImageType, ShortImageType, ShortImageType > 
		AddFilterType;
        typedef itk::DivideByConstantImageFilter< ShortImageType, int, ShortImageType >
		DivFilterType;
	typedef itk::ImageFileReader< ShortImageType > ReaderType;
	typedef itk::ImageFileWriter< ShortImageType > WriterType;
        typedef itk::GDCMImageIO ImageIOType; 
        typedef itk::MetaDataDictionary DictionaryType;
        ShortImageType::Pointer tmp;
        ShortImageType::Pointer sumImg;

        ImageIOType::Pointer gdcmImageIO = ImageIOType::New(); 
        
        ReaderType::Pointer reader = ReaderType::New();
	AddFilterType::Pointer addition = AddFilterType::New();
	DivFilterType::Pointer division = DivFilterType::New();

	division->SetConstant(nFiles);
        
        // get header information
        reader->SetFileName(inFileList[0]);
        if (isDicom) 
                reader->SetImageIO( gdcmImageIO ); 
        reader->Update();
        division->SetInput(reader->GetOutput());
	division->Update();
        sumImg = division->GetOutput();
        print_image(sumImg, "div.txt");
        print_image(sumImg, "add.txt");
        addition->SetInput1(sumImg);

        //write the output file	
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(resFile);


        if (isDicom) 
        {
                writer->SetImageIO(gdcmImageIO);
                std::cout << "dicomIO set" << std::endl;
        }


	//add all the input images
	for (int i = 1; i < nFiles; i ++)
	{
                if (i > 1)
                        addition->SetInput1(sumImg);
                        
		reader->SetFileName(inFileList[i]);
                if (isDicom) 
                        reader->SetImageIO( gdcmImageIO ); 
                try 
                {
                        reader->Update();
                }
                catch ( itk::ExceptionObject & anException ) 
                {
                        std::cerr << "\n\n** Exception in File Reader:  " << anException <<  
                                " **\n\n" << std::endl;
                        return;
                };
                tmp = reader->GetOutput();
                // print out the image before division
                std::cout << "image content before division" << std::endl;
                print_image(tmp, "readin.txt");
		// do division first
		division->SetInput(tmp);
		division->Update();
                
                // print out the image after division
                std::cout << "image content after division" << std::endl;
                tmp = division->GetOutput();
                print_image(tmp, "div.txt");

		//addition->SetInput(i, tmp);
                addition->SetInput2(tmp);
                addition->Update();
                sumImg = addition->GetOutput();

                // print out the image after addition
                std::cout << "image content after addition" << std::endl;
                print_image(sumImg, "add.txt");
	}

	writer->SetInput(sumImg);
        
        try 
        {
                writer->Update();	
        }
	catch (itk::ExceptionObject & anException)
        {
                std::cerr << "\n\n** Exception in writing result:  " << anException  << " **\n\n" << std::endl;
                return; 
        };
	
        //free memory for file name list
        for (int i = 0; i < nFiles; i ++)
                free(inFileList[i]);
        free(inFileList);
}

int main (int argc, char *argv[])
{
	// list of fnames -- to compute average
	char **fNameList; 
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
		parse_filelist(argv[1], &fNameList, &nFiles);		

		//print the input file list
		print_filelist(fNameList, nFiles);

                //check whether this is a dicom file
                isDicom = getFileExtension(fNameList[0]);

                //compute the average image
		compute_average(fNameList, nFiles, argv[2], isDicom);		
	}	
}
