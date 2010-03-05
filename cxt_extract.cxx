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
#include "cxt.h"
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

/* This function only fills in the polylines.  Structure names, id, etc. 
   will be assigned to default values if they are not already set. */
template<class T>
void
cxt_extract (Cxt_structure_list *cxt, T image, int num_structs)
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

    /* If structure names are unknown, name them, and give them 
       arbitrary id numbers */
    for (int j = cxt->num_structures; j < num_structs; j++) {
	int k = 1;
	while (cxt_find_structure_by_id (cxt, k)) k++;
	cxt_add_structure (cxt, "Unknown structure", 0, k);
    }

    /* Loop through slices */
    itSlice.SetFirstDirection(0);
    itSlice.SetSecondDirection(1);
    while (!itSlice.IsAtEnd())
    {
	typename ImageType::IndexType idx;
	UInt32Image2DType::Pointer uint32_slice;
	UCharImage2DType::Pointer uchar_slice;

	/* Make a copy of the current slice */
	idx = itSlice.GetIndex();
	uint32_slice = slice_extract (image, idx[2], (uint32_t) 0);

	and_filter->SetInput (uint32_slice);
	for (int j = 0; j < num_structs; j++)
	{
	    /* And the current slice with the mask for this structure */
	    Cxt_structure *curr_structure = &cxt->slist[j];

	    uint32_t val = (1 << curr_structure->id);
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

	    /* Run marching squares on the slice */
	    /* Note: due to an ITK bug, the contour filter cannot be 
	       "re-run" with different inputs.  Instead we should 
	       delete and create a new one for each contour. */
	    ContourFilterType::Pointer contour_filter 
		    = ContourFilterType::New();
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

	    /* Add marching squares output to cxt.  Loop through 
	       contours on this slice... */
	    for (unsigned int i = 0; 
		 i < contour_filter->GetNumberOfOutputs(); i++)
	    {
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
		/* Loop through vertices of this output contour */
		for (unsigned int k = 0; k < vertices->Size(); k++) {
		    const VertexType& vertex = vertices->ElementAt (k);
		    curr_polyline->x[k] 
			    = image->GetOrigin()[0]
			    + vertex[0] * image->GetSpacing()[0];
		    curr_polyline->y[k]
			    = image->GetOrigin()[1]
			    + vertex[1] * image->GetSpacing()[1];
		    curr_polyline->z[k] 
			    = image->GetOrigin()[2]
			    + idx[2] * image->GetSpacing()[2];
		}
	    }
	}
	itSlice.NextSlice();
    }
}

/* Explicit instantiations */
template plastimatch1_EXPORT void cxt_extract (Cxt_structure_list *cxt, UInt32ImageType::Pointer image, int num_structs);
