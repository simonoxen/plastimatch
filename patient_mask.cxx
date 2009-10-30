/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkGetAverageSliceImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryErodeImageFilter.h"

#include "itk_image.h"
#include "plm_path.h"
#include "getopt.h"

/* Thresholds for finding patient & couch */
const short T1 = -300;
const short T2 = -500;
const short T3 = -700;

class Patient_Mask_Parms {
public:
    char mha_in_fn[_MAX_PATH];
    char mha_out_fn[_MAX_PATH];
    int pt_bot;
public:
    Patient_Mask_Parms () {
	*mha_in_fn = 0;
	*mha_out_fn = 0;
	pt_bot = -1;
    }
};

void
print_usage (void)
{
    printf ("Usage: patient_mask input_mha output_mha [pt_bot]\n");
    exit (1);
}

/* Return bottom image row (indexed starting at 0) containing a patient pixel */
int
find_patient_bottom (FloatImageType::Pointer i1)
{
#if defined (commentout)
    typedef itk::ImageRegionIterator< UCharImageType > UCharIteratorType;
    typedef itk::ImageRegionIterator< DeformationFieldType > DeformationFieldIteratorType;
    DeformationFieldType::RegionType r_1 = vf->GetLargestPossibleRegion();
    UCharImageType::RegionType r_2 = mask->GetLargestPossibleRegion();

    const DeformationFieldType::IndexType& st = r_1.GetIndex();
    const DeformationFieldType::SizeType& sz = r_1.GetSize();
    const DeformationFieldType::PointType& og = vf->GetOrigin();
    const DeformationFieldType::SpacingType& sp = vf->GetSpacing();
#endif
    FloatImageType::RegionType r1 = i1->GetLargestPossibleRegion();
    const FloatImageType::SizeType& sz = r1.GetSize();
    typedef itk::ImageRegionIteratorWithIndex< FloatImageType > FloatIteratorType;

    /* In the future, these two steps can use MeanProjectionImageFilter and
	MaximumProjectionImageFilter.  But this is not implemented 
	in 2.4.1 or 3.4.0 */

    /* i3 = mean image along Z */
    /* This step can't be done with short(), because itk overflows */
    printf ("Creating slice average filter...\n");
    typedef itk::GetAverageSliceImageFilter < FloatImageType, FloatImageType > GASFilterType;
    GASFilterType::Pointer gas_filter = GASFilterType::New ();
    gas_filter->SetInput (i1);
    gas_filter->SetAveragedOutDimension (2);
    gas_filter->Update ();
    FloatImageType::Pointer i3 = gas_filter->GetOutput ();

    /* i3max = max value along each row of i3 */
    /* pt_top = top of patient, couch_bot = bottom of couch */

    float* i3max = (float*) malloc (sz[1]*sizeof(float));
    for (int i = 0; i < sz[1]; i++) {
	i3max[i] = -1000;
    }
    FloatImageType::RegionType r3 = i3->GetLargestPossibleRegion();
    FloatIteratorType it3 (i3, r3);
    for (it3.GoToBegin(); !it3.IsAtEnd(); ++it3) {
	FloatImageType::IndexType idx = it3.GetIndex();
	float pix_value = it3.Get();
	if (i3max[idx[1]] > pix_value) continue;
	i3max[idx[1]] = pix_value;
    }
    int pt_top = -1, couch_bot = -1;
    for (int i = 0; i < sz[1]; i++) {
	if (i3max[i] > T1) {
	    if (pt_top == -1) {
		pt_top = i;
	    }
	    couch_bot = i;
	}
    }

    /* pt_bot = bottom of patient */
    int pt_bot = couch_bot;
    for (int i = pt_top+1; i < couch_bot; i++) {
	if (i3max[i] < T2) {
	    pt_bot = i;
	    break;
	}
    }
    free (i3max);

    printf ("pt_top = %d, pt_bot = %d, couch_bot = %d\n", pt_top, pt_bot, couch_bot);
    return pt_bot;
}

UCharImageType::Pointer
threshold_patient (FloatImageType::Pointer i1)
{
    typedef itk::BinaryThresholdImageFilter< FloatImageType,
                                  UCharImageType > ThresholdFilterType;
    ThresholdFilterType::Pointer thresh_filter = ThresholdFilterType::New ();
    thresh_filter->SetInput (i1);
    thresh_filter->SetLowerThreshold (T3);
    // Upper threshold is max value by default
    thresh_filter->SetOutsideValue (0);
    thresh_filter->SetInsideValue (1);
    try {
	thresh_filter->Update ();
    } catch (itk::ExceptionObject & excep) {
	std::cerr << "Exception caught !" << std::endl;
	std::cerr << excep << std::endl;
    }
    return thresh_filter->GetOutput ();
}

void
remove_couch (UCharImageType::Pointer i2, int patient_bottom)
{
    typedef itk::ImageRegionIteratorWithIndex< UCharImageType > UCharIteratorType;
    UCharImageType::RegionType r2 = i2->GetLargestPossibleRegion();
    UCharIteratorType it2 (i2, r2);
    for (it2.GoToBegin(); !it2.IsAtEnd(); ++it2) {
	UCharImageType::IndexType idx = it2.GetIndex();
	if (idx[1] > patient_bottom)
	    it2.Set(0);
    }
}

UCharImageType::Pointer
erode_and_dilate (UCharImageType::Pointer i2)
{
    typedef itk::BinaryBallStructuringElement< unsigned char,3 > BallType;
    typedef itk::BinaryErodeImageFilter< UCharImageType, UCharImageType, itk::BinaryBallStructuringElement<unsigned char,3> >
	ErodeFilterType;
    typedef itk::BinaryDilateImageFilter< UCharImageType, UCharImageType, itk::BinaryBallStructuringElement<unsigned char,3> >
	DilateFilterType;
    BallType ball;
    ErodeFilterType::Pointer erode_filter = ErodeFilterType::New ();
    DilateFilterType::Pointer dilate_filter = DilateFilterType::New ();

    ball.SetRadius (2);
    ball.CreateStructuringElement ();
    
    erode_filter->SetInput (i2);
    erode_filter->SetKernel (ball);
    erode_filter->SetErodeValue (1);
    try {
	erode_filter->Update ();
    } catch (itk::ExceptionObject & excep) {
	std::cerr << "Exception caught !" << std::endl;
	std::cerr << excep << std::endl;
    }
    i2 = erode_filter->GetOutput ();
    
    dilate_filter->SetInput (i2);
    dilate_filter->SetKernel (ball);
    dilate_filter->SetDilateValue (1);
    try {
	dilate_filter->Update ();
    } catch (itk::ExceptionObject & excep) {
	std::cerr << "Exception caught !" << std::endl;
	std::cerr << excep << std::endl;
    }
    i2 = dilate_filter->GetOutput ();
    
    return i2;

}

UCharImageType::Pointer
get_largest_connected_component (UCharImageType::Pointer i2)
{
    ShortImageType::Pointer i4 = ShortImageType::New ();

    /* Identify components */
    typedef itk::ConnectedComponentImageFilter< UCharImageType, ShortImageType, UCharImageType >
	ConnectedFilterType;
    ConnectedFilterType::Pointer cc_filter = ConnectedFilterType::New ();
    cc_filter->SetInput (i2);
    try {
	cc_filter->Update ();
    } catch (itk::ExceptionObject & excep) {
	std::cerr << "Exception caught !" << std::endl;
	std::cerr << excep << std::endl;
    }
    i4 = cc_filter->GetOutput ();

    /* Sort components by size */
    typedef itk::RelabelComponentImageFilter< ShortImageType, ShortImageType >
	RelabelFilterType;
    RelabelFilterType::Pointer rel_filter = RelabelFilterType::New ();
    rel_filter->SetInput (i4);
    try {
	rel_filter->Update ();
    } catch (itk::ExceptionObject & excep) {
	std::cerr << "Exception caught !" << std::endl;
	std::cerr << excep << std::endl;
    }
    i4 = rel_filter->GetOutput ();

    /* Select largest component */
    typedef itk::BinaryThresholdImageFilter< ShortImageType,
                                  UCharImageType > LabelSelectFilterType;
    LabelSelectFilterType::Pointer ls_filter = LabelSelectFilterType::New ();
    ls_filter->SetInput (i4);
    ls_filter->SetLowerThreshold (1);
    ls_filter->SetUpperThreshold (1);
    ls_filter->SetOutsideValue (0);
    ls_filter->SetInsideValue (1);
    try {
	ls_filter->Update ();
    } catch (itk::ExceptionObject & excep) {
	std::cerr << "Exception caught !" << std::endl;
	std::cerr << excep << std::endl;
    }
    i2 = ls_filter->GetOutput ();
    return i2;
}

void
invert_image (UCharImageType::Pointer i2)
{
    typedef itk::ImageRegionIteratorWithIndex< UCharImageType > UCharIteratorType;
    UCharImageType::RegionType r2 = i2->GetLargestPossibleRegion();
    UCharIteratorType it2 (i2, r2);
    for (it2.GoToBegin(); !it2.IsAtEnd(); ++it2) {
	//UCharImageType::IndexType idx = it2.GetIndex();
	//printf ("[%d %d %d] %d %d\n", idx[2], idx[1], idx[0], it2.Get(), !it2.Get());
	it2.Set(!it2.Get());
    }
}

void
do_patient_mask (Patient_Mask_Parms* opts)
{
//    typedef itk::ImageRegionIteratorWithIndex< FloatImageType > FloatIteratorType;
    typedef itk::ImageRegionIteratorWithIndex< UCharImageType > UCharIteratorType;

    /* Load input image (i1) */
    printf ("Loading image...\n");
    FloatImageType::Pointer i1 = load_float (opts->mha_in_fn, 0);

    /* Allocate output image (i2) */
    UCharImageType::Pointer i2 = UCharImageType::New ();

    /* Find patient */
    int patient_bottom = opts->pt_bot;
    if (patient_bottom == -1) {
	patient_bottom = find_patient_bottom (i1);
    }

    /* Threshold image */
    i2 = threshold_patient (i1);

    /* Zero out the couch */
    remove_couch (i2, patient_bottom);
    //save_image (i2, "tmp0.mha");

    /* Erode and dilate */
    i2 = erode_and_dilate (i2);
    //save_image (i2, "tmp1.mha");

    /* Compute connected components */
    i2 = get_largest_connected_component (i2);
    //save_image (i2, "tmp2.mha");

    /* Invert the image */
    invert_image (i2);
    //save_image (i2, "tmp3.mha");

    /* Fill holes: Redo connected components on the (formerly) black parts */
    i2 = get_largest_connected_component (i2);

    /* Invert the image */
    invert_image (i2);

    /* Save the image */
    itk_image_save (i2, opts->mha_out_fn);
}

void
parse_args (Patient_Mask_Parms* opts, int argc, char* argv[])
{
    if (argc != 3 && argc != 4) {
	print_usage();
    }
    strncpy (opts->mha_in_fn, argv[1], _MAX_PATH);
    strncpy (opts->mha_out_fn, argv[2], _MAX_PATH);
    if (argc == 4) {
	sscanf (argv[3], "%d", &opts->pt_bot);
    }
}

int
main (int argc, char *argv[])
{
    Patient_Mask_Parms opts;
    parse_args (&opts, argc, argv);

    do_patient_mask (&opts);

    printf ("Finished!\n");
    return 0;
}
