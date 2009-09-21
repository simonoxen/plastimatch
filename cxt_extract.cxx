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
#include "itkAndConstantToImageFilter.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "cxt_io.h"
#include "cxt_extract.h"

#if defined (commentout)
static bool
debug_uchar_slice (UCharImage2DType::Pointer uchar_slice)
{
    typedef itk::ImageRegionIterator< UCharImage2DType > IteratorType;
    IteratorType it1 (uchar_slice, uchar_slice->GetBufferedRegion());

    it1.GoToBegin();
    while (!it1.IsAtEnd()) {
	unsigned char p = it1.Get ();
	if (p != 0) {
	    printf ("Got pixel: %d ", p);
	    return true;
	}
	++it1;
    }
    return false;
}

static bool
debug_uint32_slice (UInt32Image2DType::Pointer slice, uint32_t val)
{
    typedef itk::ImageRegionIterator< UInt32Image2DType > IteratorType;
    IteratorType it1 (slice, slice->GetBufferedRegion());

    it1.GoToBegin();
    while (!it1.IsAtEnd()) {
	uint32_t p = it1.Get ();
	if (p &= val) {
	    printf ("Got pixel: %d\n", p);
	    return true;
	}
	++it1;
    }
    return false;
}
#endif

template<class T>
void
cxt_extract (Cxt_structure_list *structures, T image, int num_structs)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ContourExtractor2DImageFilter<UCharImage2DType> 
	    ContourFilterType;
    typedef ContourFilterType::VertexType VertexType;
    typedef itk::ImageSliceConstIteratorWithIndex<ImageType> IteratorType;
    typedef itk::AndConstantToImageFilter<UInt32Image2DType, 
	    uint32_t, UCharImage2DType> AndFilterType;


    IteratorType itSlice (image, image->GetLargestPossibleRegion());
    typename AndFilterType::Pointer and_filter 
	    = AndFilterType::New();

    if (num_structs < 0) {
	num_structs = 32;     /* Max 32 structs in 32-bit xormap */
    }
    
    itSlice.SetFirstDirection(0);
    itSlice.SetSecondDirection(1);

    for (int j = 0; j < num_structs; j++) {
	cxt_add_structure (structures, "foo", 0, j);
    }

    while (!itSlice.IsAtEnd())
    {
	typename ImageType::IndexType idx;
	UInt32Image2DType::Pointer uint32_slice;
	UCharImage2DType::Pointer uchar_slice;

	idx = itSlice.GetIndex();
	uint32_slice = slice_extract (image, idx[2], (uint32_t) 0);

	and_filter->SetInput (uint32_slice);
		
	for (int j = 0; j < num_structs; j++) {
	    uint32_t val = (1 << j);
	    /* Note: due to an ITK bug, the contour filter cannot be 
	       "re-run" with different inputs.  Instead we should 
	       delete and create a new one for each contour. */
	    ContourFilterType::Pointer contour_filter 
		    = ContourFilterType::New();

	    and_filter->SetConstant (val);
	    try {
		and_filter->Update ();
	    }
	    catch (itk::ExceptionObject &err) {
		std::cout << "Exception during and operation." << std::endl; 
		std::cout << err << std::endl; 
		exit (1);
	    }
	    uchar_slice = and_filter->GetOutput ();

	    //	    debug_uint32_slice (uint32_slice, val);

	    //	    if (debug_uchar_slice (uchar_slice)) {
	    //		printf (" || %d %d, %d\n", idx[2], j, val);
	    //	    }
	    contour_filter->SetInput (uchar_slice);
	    contour_filter->SetContourValue (0.5);
	    try {
		contour_filter->Update();
	    }
	    catch (itk::ExceptionObject &err) {
		std::cout << "Exception during marching squares." 
			  << std::endl; 
		std::cout << err << std::endl; 
		exit (1);
	    }

	    //	    if (contour_filter->GetNumberOfOutputs() > 0) {
	    //		printf ("Got contours...\n");
	    //	    }

	    /* Add marching squares output to cxt */
	    Cxt_structure *curr_structure = &structures->slist[j];
	    for (unsigned int i = 0; i < contour_filter->GetNumberOfOutputs(); i++) {
		ContourFilterType::VertexListConstPointer vertices 
			= contour_filter->GetOutput(i)->GetVertexList();
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
	    ContourFilterType::VertexListConstPointer vertices =contour->GetOutput(i)->GetVertexList();
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
template plastimatch1_EXPORT void cxt_extract (Cxt_structure_list *structures, UInt32ImageType::Pointer image, int num_structs);
