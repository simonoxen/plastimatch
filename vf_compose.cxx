/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* This program apply transformations in sequence, first one, then the other. e.g., 
   warp image from time point #1 to #2 (vf1) and then #2 to #3 (vf2). It should be
   called with 3 input parameters. It will add up the vector fields in the first 
   2 files and write to the 3rd. More in details, the code will iterate through 
   the vector_field_1, find the target location and fetch the values in 
   vector_field_2 there, and add them up. */

/* GCS:
    compose_vector_fields vf1 vf2 vf_out
    vf_out = vf2 o vf1
    x -> x + vf2(x + vf1(x))
*/

#include <fstream>
#include <string>

#include "itkImageFileReader.h" 
#include "itkImageFileWriter.h" 
#include "itkAffineTransform.h"
#include "itkImageRegionIterator.h"
#include "itkImageToImageFilter.h"
#include "itkInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itk_image.h"
#include "itkVectorLinearInterpolateImageFunction.h"

int
main (int argc, char *argv[])
{
    if (argc!=4) {
	fprintf (stderr, "Usage: vf_compose vf_1 vf_2 vf_out\n");
	return 1;
    }

    printf ("Loading image 1...\n");
    DeformationFieldType::Pointer vf1 = load_float_field (argv[1]);
    printf ("Loading image 2...\n");
    DeformationFieldType::Pointer vf2 = load_float_field (argv[2]);
    DeformationFieldType::Pointer vf_out = DeformationFieldType::New();

    vf_out->SetRegions (vf1->GetBufferedRegion());
    vf_out->SetOrigin (vf1->GetOrigin());
    vf_out->SetSpacing (vf1->GetSpacing());
    vf_out->Allocate();

    /* No one should ever have to write code like this */
    typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
    FieldIterator vf1_it (vf1, vf1->GetBufferedRegion());
    FieldIterator vf_out_it (vf_out, vf_out->GetBufferedRegion());
    DeformationFieldType::IndexType index;
    FloatPointType point_1, point_2, point_3;
    typedef itk::Vector< float, Dimension > VectorType;
    VectorType displacement_1;
    typedef itk::VectorLinearInterpolateImageFunction < 
	DeformationFieldType, float > VectorInterpolatorType;
    VectorInterpolatorType::Pointer interpolator = VectorInterpolatorType::New();
    interpolator->SetInputImage (vf2);
    VectorInterpolatorType::OutputType displacement_2;
    VectorType displacement_3;

    vf1_it.GoToBegin();
    vf_out_it.GoToBegin();
    while (!vf1_it.IsAtEnd()) {
	index = vf1_it.GetIndex();
//	printf ("%d %d %d\n", index[0], index[1], index[2]); fflush(stdout);
	vf1->TransformIndexToPhysicalPoint (index, point_1);
	displacement_1 = vf1_it.Get ();
	for (int r = 0; r < 3; r++) {
	    point_2[r] = point_1[r] + displacement_1[r];
	}
        if (interpolator->IsInsideBuffer (point_2)) {
	    displacement_2 = interpolator->Evaluate (point_2);
	    for (int r = 0; r < 3; r++) {
		point_3[r] = point_2[r] + displacement_2[r];
		displacement_3[r] = point_3[r] - point_1[r];
	    }
	    vf_out_it.Set (displacement_3);
	} else {
	    for (int r = 0; r < 3; r++) {
		displacement_3[r] = 0.0;
	    }
	    vf_out_it.Set (displacement_3);
	}
	++vf_out_it;
	++vf1_it;
    }

    itk_image_save (vf_out, argv[3]);

    return 0;
}

