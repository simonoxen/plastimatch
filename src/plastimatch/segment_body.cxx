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
#include "itk_image_save.h"
#include "plm_path.h"
#include "plm_image_header.h"
#include "resample_mha.h"
#include "segment_body.h"

/* Thresholds for finding patient & couch */
const short T1 = -300;
const short T2 = -500;
const short T3 = -1000;

/* Return bottom image row (indexed starting at 0) containing a patient pixel */
int
Segment_body::find_patient_bottom (FloatImageType::Pointer i1)
{
    FloatImageType::RegionType r1 = i1->GetLargestPossibleRegion();
    const FloatImageType::SizeType& sz = r1.GetSize();
    typedef itk::ImageRegionIteratorWithIndex< 
	FloatImageType > FloatIteratorType;

    /* In the future, these two steps can use MeanProjectionImageFilter and
	MaximumProjectionImageFilter.  But this is not implemented 
	in 2.4.1 or 3.4.0 */

    /* i3 = mean image along Z */
    /* This step can't be done with short(), because itk overflows */
    typedef itk::GetAverageSliceImageFilter < 
	FloatImageType, FloatImageType > GASFilterType;
    GASFilterType::Pointer gas_filter = GASFilterType::New ();
    gas_filter->SetInput (i1);
    gas_filter->SetAveragedOutDimension (2);
    gas_filter->Update ();
    FloatImageType::Pointer i3 = gas_filter->GetOutput ();

    /* i3max = max value along each row of i3 */
    /* pt_top = top of patient, couch_bot = bottom of couch */

    float* i3max = (float*) malloc (sz[1]*sizeof(float));
    for (unsigned long i = 0; i < sz[1]; i++) {
	i3max[i] = m_lower_threshold;
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
    for (unsigned long i = 0; i < sz[1]; i++) {
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
Segment_body::threshold_patient (FloatImageType::Pointer i1)
{
    typedef itk::BinaryThresholdImageFilter< 
	FloatImageType, UCharImageType > ThresholdFilterType;
    ThresholdFilterType::Pointer thresh_filter = ThresholdFilterType::New ();
    thresh_filter->SetInput (i1);
    thresh_filter->SetLowerThreshold (m_lower_threshold);
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
    typedef itk::ImageRegionIteratorWithIndex< 
	UCharImageType > UCharIteratorType;
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
    typedef itk::BinaryErodeImageFilter< 
	UCharImageType, UCharImageType, 
	itk::BinaryBallStructuringElement<unsigned char,3> > ErodeFilterType;
    typedef itk::BinaryDilateImageFilter< 
	UCharImageType, UCharImageType, 
	itk::BinaryBallStructuringElement<unsigned char,3> > DilateFilterType;
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
    typedef itk::ConnectedComponentImageFilter< 
    UCharImageType, ShortImageType, UCharImageType > ConnectedFilterType;
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

    int num_vox = 
	i4->GetLargestPossibleRegion().GetSize()[0] *
	i4->GetLargestPossibleRegion().GetSize()[1] *
	i4->GetLargestPossibleRegion().GetSize()[2];
    int label_upper_thresh = 1;
    float ccs_percent_thresh = 0.05;
    for (unsigned int ccs = 0; 
	 ccs < rel_filter->GetSizeOfObjectsInPixels().size();
	 ++ccs) 
    {
	float cc_pct = 
	    (float) rel_filter->GetSizeOfObjectsInPixels()[ccs] / num_vox;
	if (cc_pct > ccs_percent_thresh) {
	    label_upper_thresh = ccs + 1;
	    printf ("CC %d has size %d (%f,%f)\n", ccs, 
		rel_filter->GetSizeOfObjectsInPixels()[ccs],
		cc_pct,
		ccs_percent_thresh
	    );
	} else {
	    break;
	}
    }

    /* Select largest N components */
    typedef itk::BinaryThresholdImageFilter< ShortImageType,
	UCharImageType > LabelSelectFilterType;
    LabelSelectFilterType::Pointer ls_filter = LabelSelectFilterType::New ();
    ls_filter->SetInput (i4);
    ls_filter->SetLowerThreshold (1);
    ls_filter->SetUpperThreshold (label_upper_thresh);
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
    typedef itk::ImageRegionIteratorWithIndex< 
	UCharImageType > UCharIteratorType;
    UCharImageType::RegionType r2 = i2->GetLargestPossibleRegion();
    UCharIteratorType it2 (i2, r2);
    for (it2.GoToBegin(); !it2.IsAtEnd(); ++it2) {
	it2.Set(!it2.Get());
    }
}

void
Segment_body::do_segmentation ()
{
    /* Convert input to float */
    FloatImageType::Pointer i1 = this->img_in.itk_float();

    /* Reduce resolution to at most 5 mm voxels */
    if (m_fast) {
	Plm_image_header pih;
	bool need_resample = false;
	ImageRegionType::SizeType itk_size;
	pih.set_from_plm_image (&this->img_in);
	for (int d = 0; d < 3; d++) {
	    if (pih.m_spacing[d] < 5.0) {
		itk_size[d] = (int) floor(pih.Size(d) * pih.m_spacing[d] / 5.0);
		pih.m_origin[d] += (5.0 - pih.m_spacing[d]) / 2;
		pih.m_spacing[d] = 5.0;
		need_resample = true;
	    } else {
		itk_size[d] = pih.Size(d);
	    }
	}
	if (need_resample) {
	    printf ("Resampling image\n");
	    pih.m_region.SetSize (itk_size);
	    i1 = resample_image (i1, &pih, -1000, 1);
	    if (m_debug) {
		itk_image_save (i1, "0_resample.nrrd");
	    }
	}
    }

    /* Allocate output image (i2) */
    UCharImageType::Pointer i2 = UCharImageType::New ();

    /* Find patient */
    int patient_bottom;
    if (this->m_bot_given) {
	patient_bottom = this->m_bot;
    } else {
	printf ("find_patient_bottom\n");
	patient_bottom = find_patient_bottom (i1);
    }

    /* Threshold image */
    printf ("threshold\n");
    i2 = this->threshold_patient (i1);

    /* Zero out the couch */
    printf ("remove_couch\n");
    remove_couch (i2, patient_bottom);
    if (m_debug) {
	itk_image_save (i2, "1_remove_couch.nrrd");
    }

    /* Erode and dilate */
    printf ("erode_and_dilate\n");
    i2 = erode_and_dilate (i2);

    /* Compute connected components */
    printf ("get_largest_connected_component\n");
    i2 = get_largest_connected_component (i2);

    /* Invert the image */
    printf ("invert\n");
    invert_image (i2);
    if (m_debug) {
	itk_image_save (i2, "2_largest_cc.nrrd");
    }

    /* Fill holes: Redo connected components on the (formerly) black parts */
    printf ("get_largest_connected_component\n");
    i2 = get_largest_connected_component (i2);

    if (m_debug) {
	itk_image_save (i2, "3_re_invert.nrrd");
    }

    /* Invert the image */
    printf ("invert\n");
    invert_image (i2);

    /* Return image to caller */
    printf ("return\n");
    this->img_out.set_itk (i2);
}
