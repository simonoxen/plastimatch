/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: DanielssonDistanceMapImageFilter.cxx,v $
  Language:  C++
  Date:      $Date: 2009-03-16 21:52:48 $
  Version:   $Revision: 1.29 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "itk_image.h"
//#include "itkDanielssonDistanceMapImageFilter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"

typedef UCharImageType InputImageType;
typedef FloatImageType OutputImageType; 
typedef itk::ImageFileReader< InputImageType  >  ReaderType;
typedef itk::ImageFileWriter< OutputImageType >  WriterType;
typedef itk::SignedMaurerDistanceMapImageFilter< InputImageType, OutputImageType >  FilterType;
typedef itk::RescaleIntensityImageFilter<OutputImageType, OutputImageType > RescalerType;

void print_usage (void)
{
	std::cerr << "Usage: distance_map inputImageFile outputDistanceMapImageFile";
    std::cerr << std::endl;  
	exit (-1);
}

int main( int argc, char * argv[] )
{
	InputImageType::Pointer reference=InputImageType::New();
	//ImgType::Pointer warped=ImgType::New();
	FilterType::Pointer filter = FilterType::New();
	RescalerType::Pointer scaler = RescalerType::New();
  
  if( argc < 3 )
	  print_usage();

  filter->UseImageSpacingOn();
  filter->SquaredDistanceOff();


  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  reader->SetFileName( argv[1] );
  writer->SetFileName( argv[2] );
  filter->SetInput( reader->GetOutput() );

  filter->SetInput( reader->GetOutput() );

  scaler->SetInput( filter->GetOutput() );
  writer->SetInput( scaler->GetOutput() );
  scaler->SetOutputMaximum( 65535L );
  scaler->SetOutputMinimum(     0L );
  
  writer->Update();
}
