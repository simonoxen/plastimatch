/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/*  Modify the input image, setting the pixel values within the mask. 
    Modified by Ziji Wu, 3/2006, to do the vector field as well
*/
#include <time.h>
#include "plm_config.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkImageRegionIterator.h"

#include "itk_image.h"

typedef UCharImageType MaskImageType;
typedef ShortImageType RealImageType;

void
merge_pixels(ShortImageType::Pointer im_out, ShortImageType::Pointer im_1, 
	     UCharImageType::Pointer im_2, int mask_value)
{
    typedef itk::ImageRegionIterator< UCharImageType > UCharIteratorType;
    typedef itk::ImageRegionIterator< ShortImageType > ShortIteratorType;
    ShortImageType::RegionType r_1 = im_1->GetLargestPossibleRegion();
    UCharImageType::RegionType r_2 = im_2->GetLargestPossibleRegion();

    //const ShortImageType::IndexType& st = r_1.GetIndex();
    //const ShortImageType::SizeType& sz = r_1.GetSize();
    //const InputImageType::SizeType& sz = image->GetLargestPossibleRegion().GetSize();
    const ShortImageType::PointType& og = im_1->GetOrigin();
    const ShortImageType::SpacingType& sp = im_1->GetSpacing();
    
    im_out->SetRegions(r_1);
    im_out->SetOrigin(og);
    im_out->SetSpacing(sp);
    im_out->Allocate();

    ShortIteratorType it_1 (im_1, r_1);
    UCharIteratorType it_2 (im_2, r_2);
    ShortIteratorType it_out (im_out, r_1);

    for (it_1.GoToBegin(); !it_1.IsAtEnd(); ++it_1,++it_2,++it_out) {
	short p1 = it_1.Get();
	unsigned char p2 = it_2.Get();
	if (p2 > 0) {
	    it_out.Set (p1);
	} else {
	    it_out.Set (mask_value);
	}
    }
}

void
mask_vf(DeformationFieldType::Pointer vf_out, DeformationFieldType::Pointer vf, 
	     UCharImageType::Pointer mask, float mask_value[3])
{
    typedef itk::ImageRegionIterator< UCharImageType > UCharIteratorType;
    typedef itk::ImageRegionIterator< DeformationFieldType > DeformationFieldIteratorType;
    DeformationFieldType::RegionType r_1 = vf->GetLargestPossibleRegion();
    UCharImageType::RegionType r_2 = mask->GetLargestPossibleRegion();

    //const DeformationFieldType::IndexType& st = r_1.GetIndex();
    //const DeformationFieldType::SizeType& sz = r_1.GetSize();
    const DeformationFieldType::PointType& og = vf->GetOrigin();
    const DeformationFieldType::SpacingType& sp = vf->GetSpacing();
    
    vf_out->SetRegions(r_1);
    vf_out->SetOrigin(og);
    vf_out->SetSpacing(sp);
    vf_out->Allocate();

    DeformationFieldIteratorType it_1 (vf, r_1);
    UCharIteratorType it_2 (mask, r_2);
    DeformationFieldIteratorType it_out (vf_out, r_1);

    for (it_1.GoToBegin(); !it_1.IsAtEnd(); ++it_1,++it_2,++it_out) {
	itk::Vector<float,3> p1 = it_1.Get();
	unsigned char p2 = it_2.Get();
	if (p2 > 0) {
	    it_out.Set (p1);
	} else {
	    it_out.Set (mask_value);
	}
    }
}

int
main(int argc, char *argv[])
{
    if ((argc != 5) && (argc != 7)) {
	std::cerr << "Wrong Parameters " << std::endl;
	std::cerr << "Usage: ";
	std::cerr << "input_image mask_image mask_value output"<< std::endl;
	return 1;
    }

    if (argc == 5) { // masking an image volume
	int mask_value;
	typedef itk::ImageFileWriter < ShortImageType > WriterType;

	ShortImageType::Pointer im_1 = RealImageType::New();
	UCharImageType::Pointer im_2 = MaskImageType::New();
	ShortImageType::Pointer im_out = ShortImageType::New();

	printf ("Loading...\n");
	im_1 = itk_image_load_short (argv[1], 0);
	im_2 = itk_image_load_uchar (argv[2], 0);

	sscanf (argv[3], "%d", &mask_value);
	printf ("Setting mask value to %d\n", mask_value);
	merge_pixels (im_out, im_1, im_2, mask_value);

	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(argv[4]);
	writer->SetInput(im_out);
	writer->Update();
    } else { // masking a vector field
	float mask_value[3];

	typedef itk::ImageFileWriter < DeformationFieldType > WriterType;

	DeformationFieldType::Pointer vf = DeformationFieldType::New();
	UCharImageType::Pointer mask = MaskImageType::New();
	DeformationFieldType::Pointer vf_out = DeformationFieldType::New();

	printf ("Loading...\n");
	vf = itk_image_load_float_field (argv[1]);
	mask = itk_image_load_uchar (argv[2], 0);

	printf ("Masking...\n");

	sscanf (argv[3], "%f", &(mask_value[0]));
	sscanf (argv[4], "%f", &(mask_value[1]));
	sscanf (argv[5], "%f", &(mask_value[2]));

	mask_vf(vf_out, vf, mask, mask_value);

	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(argv[6]);
	writer->SetInput(vf_out);
	writer->Update();
    }

    printf ("Finished!\n");
    return 0;
}
