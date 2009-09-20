/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_int.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkContourExtractor2DImageFilter.h"
#include "itkImage.h"
#include "itk_image.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "slice_extract.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "cxt_io.h"
#include "cxt_extract.h"

template<class T>
void
cxt_extract (Cxt_structure_list *structures, T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ContourExtractor2DImageFilter<UCharImage2DType> 
	    ContourType;
    typedef ContourType::VertexType VertexType;
    typedef itk::ImageSliceConstIteratorWithIndex<ImageType> IteratorType;
    typedef itk::BinaryThresholdImageFilter<UInt32Image2DType, 
	    UCharImage2DType> ThresholdFilterType;

    IteratorType itSlice (image, image->GetLargestPossibleRegion());
    typename ThresholdFilterType::Pointer threshold_filter 
	    = ThresholdFilterType::New();
    ContourType::Pointer contour = ContourType::New();
    
    itSlice.SetFirstDirection(0);
    itSlice.SetSecondDirection(1);

    for (int j = 0; j < 10; j++) {
	cxt_add_structure (structures, "foo", j);
    }

    while (!itSlice.IsAtEnd())
    {
	typename ImageType::IndexType idx;
	UInt32Image2DType::Pointer uint32_slice;
	UCharImage2DType::Pointer uchar_slice;

	idx = itSlice.GetIndex();
	uint32_slice = slice_extract (image, idx[2], (uint32_t) 0);

	threshold_filter->SetInput (uint32_slice);
	threshold_filter->SetOutsideValue (0);
	threshold_filter->SetInsideValue (1);
		
	for (int j = 0; j < 10; j++) {
	    printf("%2ld%c\n", idx[2], 'a' + j);
	    threshold_filter->SetOutsideValue (0);
	    threshold_filter->SetInsideValue (1);
	    threshold_filter->SetUpperThreshold (j);
	    threshold_filter->SetLowerThreshold (j);
	    try {
		threshold_filter->Update ();
	    }
	    catch (itk::ExceptionObject &err) {
		std::cout << "Exception during threshold." << std::endl; 
		std::cout << err << std::endl; 
		return;
	    }
	    uchar_slice = threshold_filter->GetOutput ();
	    contour->SetInput (uchar_slice);
	    contour->SetContourValue (0.5);
	    try {
		contour->Update();
	    }
	    catch (itk::ExceptionObject &err) {
		std::cout << "Exception during marching squares." 
			  << std::endl; 
		std::cout << err << std::endl; 
		return;
	    }

	    /* Add marching squares output to cxt */
	    Cxt_structure *curr_structure = &structures->slist[j];
	    for (unsigned int i = 0; i < contour->GetNumberOfOutputs(); i++) {
		ContourType::VertexListConstPointer vertices 
			= contour->GetOutput(i)->GetVertexList();
		Cxt_polyline *curr_polyline 
			= cxt_add_polyline (curr_structure);

		curr_polyline->num_vertices = vertices->Size();
		curr_polyline->x = (float*) 
			malloc (vertices->Size() * sizeof(float));
		curr_polyline->y = (float*) 
			malloc (vertices->Size() * sizeof(float));
		curr_polyline->z = (float*) 
			malloc (vertices->Size() * sizeof(float));
		for (unsigned int k = 0; k < vertices->Size(); k++) {
		    const VertexType& vertex = vertices->ElementAt (k);
		    curr_polyline->x[k] = vertex[0];
		    curr_polyline->y[k] = vertex[1];
		    curr_polyline->z[k] = idx[2];
		}
	    }
	}

#if defined (commentout)		
	//std::cout << "NR OUTPUTS:"<<contour->GetNumberOfOutputs() << std::endl; 
	//system("PAUSE");
	for(unsigned int i = 0; i < contour->GetNumberOfOutputs(); i++)
	{
	    ContourType::VertexListConstPointer vertices =contour->GetOutput(i)->GetVertexList();
	    /*fprintf(fp,"%s %d%s%d\n","Contour",k[2],".",i);*/
	    /*fprintf(fp,"%d%s%d\n",k[2],".",i);*/
	    //fprintf(fp,"\n");
	    fprintf(fp,"%s %s %s\n","NaN","NaN","NaN");
	    fprintf(file,"%s %s %s\n","NaN","NaN","NaN");
	    for(unsigned int j = 0; j < vertices->Size(); j++)
	    {
		const VertexType& vertex = vertices->ElementAt(j);
					
		fprintf(fp,"%.3f %.3f %2d\n",vertex[0],vertex[1],k[2]);
		fprintf(file,"%.3f %.3f %.3f \n",vertex[0]*image->GetSpacing()[0]+image->GetSpacing()[3],vertex[1]*image->GetSpacing()[1]+image->GetSpacing()[4],k[2]*image->GetSpacing()[2]+image->GetSpacing()[5]);
		
		//fprintf(fp,"%.3f %.3f %2d\n",vertex[0],vertex[1],k[2]*image->GetSpacing()[2]);
		//std::cout << vertex[0] <<" "<<vertex[1]<<" "<<k[2]<<std::endl;
					

	    }
	    //system("PAUSE");
	}
#endif
	itSlice.NextSlice();
    }
}

/* Explicit instantiations */
template plastimatch1_EXPORT void cxt_extract (Cxt_structure_list *structures, UInt32ImageType::Pointer image);
