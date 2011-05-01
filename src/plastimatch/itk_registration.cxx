/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <string.h>
#include "itkCenteredTransformInitializer.h"
#include "itkImageMaskSpatialObject.h"
#include "itkImageRegistrationMethod.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkMutualInformationImageToImageMetric.h"
#include "itkRegularStepGradientDescentOptimizer.h"

//#define USE_GCS_METRIC 1

#if defined(ITK_USE_OPTIMIZED_REGISTRATION_METHODS)
#include "itkOptMattesMutualInformationImageToImageMetric.h"
#include "itkOptMeanSquaresImageToImageMetric.h"
#else
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetric.h"
#if (USE_GCS_METRIC)
#include "gcs_metric.h"
#else
#endif
#endif

#include "itk_demons.h"
#include "itk_image.h"
#include "itk_optim.h"
#include "itk_warp.h"
#include "plm_image_header.h"
#include "plm_int.h"
#include "plm_parms.h"
#include "plm_timer.h"
#include "resample_mha.h"
#include "xform.h"

#if !defined (ITK_USE_OPTIMIZED_REGISTRATION_METHODS) && defined (USE_GCS_METRIC)
typedef itk::GCSMetric <
    FloatImageType, FloatImageType > MSEMetricType;
#else
typedef itk::MeanSquaresImageToImageMetric <
    FloatImageType, FloatImageType > MSEMetricType;
#endif
typedef itk::MutualInformationImageToImageMetric <
    FloatImageType, FloatImageType > MIMetricType;
typedef itk::MattesMutualInformationImageToImageMetric <
    FloatImageType, FloatImageType > MattesMIMetricType;

typedef itk::ImageMaskSpatialObject< 3 > Mask_SOType;

typedef itk::LinearInterpolateImageFunction <
    FloatImageType, double >InterpolatorType;

class Optimization_Observer : public itk::Command
{
public:
    typedef Optimization_Observer Self;
    typedef itk::Command Superclass;
    typedef itk::SmartPointer < Self > Pointer;
    itkNewMacro(Self);
protected:
    Optimization_Observer() {
	m_stage = 0;
	plm_timer_start (&timer);
    };
public:
    Stage_parms* m_stage;
    RegistrationType::Pointer m_registration;
    double last_value;
    Timer timer;

    void Set_Stage_parms (RegistrationType::Pointer registration,
	Stage_parms* stage) {
	m_registration = registration;
	m_stage = stage;
    }

    void Execute (itk::Object * caller, const itk::EventObject & event) {
        Execute((const itk::Object *) caller, event);
	plm_timer_start (&timer);
    }

    void
    Execute (const itk::Object * object, const itk::EventObject & event)
    {
	if (typeid(event) == typeid(itk::StartEvent)) {
	    last_value = -1.0;
	    std::cout << "StartEvent: ";
	    if (m_stage->xform_type != STAGE_TRANSFORM_BSPLINE) {
		std::cout << optimizer_get_current_position (
		    m_registration, m_stage);
	    }
	    std::cout << std::endl;
	    plm_timer_start (&timer);
	}
	else if (typeid(event) == typeid(itk::InitializeEvent)) {
	    std::cout << "InitializeEvent: ";
	    std::cout << std::endl;
	    plm_timer_start (&timer);
	}
	else if (typeid(event) == typeid(itk::EndEvent)) {
	    std::cout << "EndEvent: ";
	    if (m_stage->xform_type != STAGE_TRANSFORM_BSPLINE) {
		std::cout << optimizer_get_current_position (
		    m_registration, m_stage);
		std::cout << std::endl;
	    }
	    std::cout << std::endl;
	}
	else if (typeid(event) 
	    == typeid(itk::FunctionEvaluationIterationEvent))
	{
	    std::cout << "FunctionEvaluationIterationEvent\n";
	}
	else if (typeid(event) 
	    == typeid(itk::FunctionAndGradientEvaluationIterationEvent))
	{
	    double duration;
	    
	    std::cout << "VAL+GRAD ";
	    double val = optimizer_get_value(m_registration, m_stage);

	    duration = plm_timer_report (&timer);
	    printf ("%6.3f [%6.3f secs]", val, duration);
	    std::cout << std::endl;
	    plm_timer_start (&timer);
	}
	else if (typeid(event) == typeid(itk::IterationEvent)) {
	    std::cout << "IterationEvent: ";
	    int it = optimizer_get_current_iteration(m_registration, m_stage);
	    double val = optimizer_get_value(m_registration, m_stage);
	    double ss = optimizer_get_step_length(m_registration, m_stage);
	    
	    printf ("%3d %10.2f %5.2f ", it, val, ss);

	    if (m_stage->xform_type != STAGE_TRANSFORM_BSPLINE) {
		std::cout << optimizer_get_current_position (
		    m_registration, m_stage);
	    }

	    if (last_value >= 0.0) {
		double diff = fabs(last_value - val);
		if (it >= m_stage->min_its && diff < m_stage->convergence_tol) {
		    printf (" %10.2f (tol)", diff);
		    /* calling optimizer_set_max_iterations () doesn't 
		       seem to always stop rsg. */

		    if (m_stage->optim_type == OPTIMIZATION_RSG) {
			typedef itk::RegularStepGradientDescentOptimizer * OptimizerPointer;
			OptimizerPointer optimizer = dynamic_cast< OptimizerPointer >(m_registration->GetOptimizer());
			optimizer->StopOptimization();
		    } else {
			optimizer_set_max_iterations (m_registration, m_stage, 1);
		    }
		} else {
		    printf (" %10.2f", diff);
		}
	    }
	    last_value = val;
	    std::cout << std::endl;
	}
	else if (typeid(event) == typeid(itk::ProgressEvent)) {
	    std::cout << "ProgressEvent: ";
	    if (m_stage->xform_type != STAGE_TRANSFORM_BSPLINE) {
		std::cout << optimizer_get_current_position (m_registration, m_stage);
	    }
	    std::cout << std::endl;
	}
	else {
	    std::cout << "Unknown event type." << std::endl;
	    event.Print(std::cout);
	}
    }
};

void
set_metric (RegistrationType::Pointer registration,
	    Stage_parms* stage)
{
    switch (stage->metric_type) {
    case METRIC_MSE:
	{
	    MSEMetricType::Pointer metric = MSEMetricType::New();
	    registration->SetMetric(metric);
	}
	break;
    case METRIC_MI:
	{
	    /*  The metric requires a number of parameters to be
		selected, including the standard deviation of the
		Gaussian kernel for the fixed image density estimate,
		the standard deviation of the kernel for the moving
		image density and the number of samples use to compute
		the densities and entropy values. Details on the
		concepts behind the computation of the metric can be
		found in Section \ref{sec:MutualInformationMetric}.
		Experience has shown that a kernel standard deviation
		of $0.4$ works well for images which have been
		normalized to a mean of zero and unit variance.  We
		will follow this empirical rule in this example. */
	    MIMetricType::Pointer metric = MIMetricType::New();
	    metric->SetFixedImageStandardDeviation(  0.4 );
	    metric->SetMovingImageStandardDeviation( 0.4 );
	    registration->SetMetric(metric);
	}
	break;
    case METRIC_MI_MATTES:
	{
	    /*  The metric requires two parameters to be selected: the 
		number of bins used to compute the entropy and the
		number of spatial samples used to compute the density
		estimates. In typical application, 50 histogram bins
		are sufficient and the metric is relatively
		insensitive to changes in the number of bins. The
		number of spatial samples to be used depends on the
		content of the image. If the images are smooth and do
		not contain much detail, then using approximately $1$
		percent of the pixels will do. On the other hand, if
		the images are detailed, it may be necessary to use a
		much higher proportion, such as $20$ percent. */
	    MattesMIMetricType::Pointer metric = MattesMIMetricType::New();
	    metric->SetNumberOfHistogramBins(stage->mi_histogram_bins);
	    metric->SetNumberOfSpatialSamples(stage->mi_num_spatial_samples);
	    registration->SetMetric(metric);
	}
	break;
    default:
	print_and_exit ("Error: metric is not implemented");
	break;
    }
}

void
set_mask_images (RegistrationType::Pointer registration,
		 Registration_Data* regd,
		 Stage_parms* stage)
{
    if (regd->fixed_mask) {
	Mask_SOType::Pointer mask_so = Mask_SOType::New();
	mask_so->SetImage(regd->fixed_mask);
	mask_so->Update();
	registration->GetMetric()->SetFixedImageMask (mask_so);
    }
    if (regd->moving_mask) {
	Mask_SOType::Pointer mask_so = Mask_SOType::New();
	mask_so->SetImage(regd->moving_mask);
	mask_so->Update();
	registration->GetMetric()->SetMovingImageMask (mask_so);
    }
}

/* This helps speed up the registration, by setting the bounding box to the 
   smallest size needed.  To find the bounding box, either use the extent 
   of the fixed_mask (if one is used), or by eliminating 
   excess air by thresholding
*/
void
set_fixed_image_region_new_unfinished (
    RegistrationType::Pointer registration,
    Registration_Data* regd,
    Stage_parms* stage
)
{
    FloatImageType::RegionType valid_region;
    FloatImageType::RegionType::IndexType valid_index;
    FloatImageType::RegionType::SizeType valid_size;

    FloatImageType::ConstPointer fi = static_cast < FloatImageType::ConstPointer > (registration->GetFixedImage());

    for (int d = 0; d < 3; d++) {
	float ori = regd->fixed_region_origin[d] + regd->fixed_region.GetIndex()[d] * regd->fixed_region_spacing[d];
	int idx = (int) floor (ori - (fi->GetOrigin()[d] 
				      - 0.5 * fi->GetSpacing()[d]) 
			       / fi->GetSpacing()[d]);
	if (idx < 0) {
	    fprintf (stderr, "set_fixed_image_region conversion error.\n");
	    exit (-1);
	}
	float last_pix_center = ori + (regd->fixed_region.GetSize()[d]-1) 
		* regd->fixed_region_spacing[d];
	int siz = (int) floor (last_pix_center - (fi->GetOrigin()[d] 
						  - 0.5 * fi->GetSpacing()[d]) 
			       / fi->GetSpacing()[d]);
	siz = siz - idx + 1;
	valid_index[d] = idx;
	valid_size[d] = siz;
    }

    valid_region.SetIndex (valid_index);
    valid_region.SetSize (valid_size);
    registration->SetFixedImageRegion (valid_region);
}

void
set_fixed_image_region (RegistrationType::Pointer registration,
			Registration_Data* regd,
			Stage_parms* stage)
{
    int use_magic_value = 0;
    if (regd->fixed_mask) {
	FloatImageType::RegionType valid_region;
	FloatImageType::RegionType::IndexType valid_index;
	FloatImageType::RegionType::SizeType valid_size;
	valid_index[0] = 0;
	valid_index[1] = 0;
	valid_index[2] = 0;
	valid_size[0] = 1;
	valid_size[1] = 1;
	valid_size[2] = 1;

	/* Search for bounding box of fixed mask */
	typedef Mask_SOType* Mask_SOPointer;
	Mask_SOPointer so = (Mask_SOPointer) registration->GetMetric()->GetFixedImageMask();

	typedef itk::ImageRegionConstIteratorWithIndex< UCharImageType > IteratorType;
	UCharImageType::RegionType region = registration->GetFixedImage()->GetLargestPossibleRegion();
	IteratorType it (so->GetImage(), region);

	int first = 1;
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    unsigned char c = it.Get();
	    if (c) {
		UCharImageType::RegionType::IndexType idx = it.GetIndex();
		if (first) {
		    first = 0;
		    valid_index = idx;
		    valid_size[0] = 1;
		    valid_size[1] = 1;
		    valid_size[2] = 1;
		} else {
		    int updated = 0;
		    for (int i = 0; i < 3; i++) {
			if (valid_index[i] > idx[i]) {
			    valid_size[i] += valid_index[i] - idx[i];
			    valid_index[i] = idx[i];
			    updated = 1;
			}
			if (idx[i] - valid_index[i] >= (long) valid_size[i]) {
			    valid_size[i] = idx[i] - valid_index[i] + 1;
			    updated = 1;
			}
		    }
		}
	    }
	}
	valid_region.SetIndex(valid_index);
	valid_region.SetSize(valid_size);
	registration->SetFixedImageRegion(valid_region);
    } else if (use_magic_value) {
	FloatImageType::RegionType valid_region;
	FloatImageType::RegionType::IndexType valid_index;
	FloatImageType::RegionType::SizeType valid_size;

	/* Search for bounding box of patient */
	typedef itk::ImageRegionConstIteratorWithIndex<FloatImageType> IteratorType;
	FloatImageType::RegionType region = registration->GetFixedImage()->GetLargestPossibleRegion();
	IteratorType it (registration->GetFixedImage(), region);

	int first = 1;
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float c = it.Get();
	    if (c > stage->background_max) {
		FloatImageType::RegionType::IndexType idx = it.GetIndex();
		if (first) {
		    first = 0;
		    valid_index = idx;
		    valid_size[0] = 1;
		    valid_size[1] = 1;
		    valid_size[2] = 1;
		} else {
		    int updated = 0;
		    for (int i = 0; i < 3; i++) {
			if (valid_index[i] > idx[i]) {
			    valid_size[i] += valid_index[i] - idx[i];
			    valid_index[i] = idx[i];
			    updated = 1;
			}
			if (idx[i] - valid_index[i] >= (long) valid_size[i]) {
			    valid_size[i] = idx[i] - valid_index[i] + 1;
			    updated = 1;
			}
		    }
		}
	    }
	}
	/* Try to include a margin of at least one air pixel everywhere */
	for (int i = 0; i < 3; i++) {
	    if (valid_index[i] > 0) {
		valid_index[i]--;
		valid_size[i]++;
	    }
	    if (valid_size[i] + valid_index[i] < registration->GetFixedImage()->GetLargestPossibleRegion().GetSize()[i]) {
		valid_size[i]++;
	    }
	}
	valid_region.SetIndex(valid_index);
	valid_region.SetSize(valid_size);
	registration->SetFixedImageRegion(valid_region);
    } else {
	registration->SetFixedImageRegion(registration->GetFixedImage()->GetLargestPossibleRegion());
    }
}

template<class ImgP>
void
show_image_stats (ImgP image)
{
    typedef typename ImgP::ObjectType Img;

    const typename Img::SizeType& sz = image->GetLargestPossibleRegion().GetSize();
    const typename Img::PointType& ori = image->GetOrigin();
    const typename Img::SpacingType& sp = image->GetSpacing();
    const typename Img::DirectionType& di = image->GetDirection();

    printf ("Origin = %g %g %g\n", ori[0], ori[1], ori[2]);
    printf ("Spacing = %g %g %g\n", sp[0], sp[1], sp[2]);
    std::cout << "Size = " << sz[0] << " " << sz[1] << " " << sz[2] << std::endl;
    printf ("Direction Cosines =\n");
    for (unsigned int d1 = 0; d1 < 3; d1++) {
	for (unsigned int d2 = 0; d2 < 3; d2++) {
	    printf (" %g", di[d1][d2]);
	}
	printf ("\n");
    }
}

void
show_stats (RegistrationType::Pointer registration)
{
    show_image_stats(static_cast < FloatImageType::ConstPointer > (registration->GetFixedImage()));
    show_image_stats(static_cast < FloatImageType::ConstPointer > (registration->GetMovingImage()));
}

void
set_transform_translation (RegistrationType::Pointer registration,
			Xform *xf_out,
			Xform *xf_in,
			Stage_parms* stage)
{
    Plm_image_header pih;
    pih.set_from_itk_image (registration->GetFixedImage());
    xform_to_trn (xf_out, xf_in, &pih);
    registration->SetTransform (xf_out->get_trn());
}

void
set_transform_versor (RegistrationType::Pointer registration,
			Xform *xf_out,
			Xform *xf_in,
			Stage_parms* stage)
{
    Plm_image_header pih;
    pih.set_from_itk_image (registration->GetFixedImage());
    xform_to_vrs (xf_out, xf_in, &pih);
    registration->SetTransform (xf_out->get_vrs());
}

void
set_transform_quaternion (
    RegistrationType::Pointer registration,
    Xform *xf_out,
    Xform *xf_in,
    Stage_parms* stage)
{
    Plm_image_header pih;
    pih.set_from_itk_image (registration->GetFixedImage());
    xform_to_quat (xf_out, xf_in, &pih);
    registration->SetTransform (xf_out->get_quat());
}

void
set_transform_affine (RegistrationType::Pointer registration,
			Xform *xf_out,
			Xform *xf_in,
			Stage_parms* stage)
{
    Plm_image_header pih;
    pih.set_from_itk_image (registration->GetFixedImage());
    xform_to_aff (xf_out, xf_in, &pih);
    registration->SetTransform (xf_out->get_aff());
}

void
set_transform_bspline (
    RegistrationType::Pointer registration,
    Xform *xf_out,
    Xform *xf_in,
    Stage_parms* stage
)
{
    Plm_image_header pih;
    pih.set_from_itk_image (registration->GetFixedImage());		    

    /* GCS FIX: Need to set ROI from registration->GetFixedImageRegion(), */
    xform_to_itk_bsp (xf_out, xf_in, &pih, stage->grid_spac);
    registration->SetTransform (xf_out->get_itk_bsp());

//    registration->SetInitialTransformParameters (
//	xf_out->get_itk_bsp()->GetParameters());

}

void
set_transform (
    RegistrationType::Pointer registration,
    Xform *xf_out,
    Xform *xf_in,
    Stage_parms* stage
)
{
    xf_out->clear();
    switch (stage->xform_type) {
    case STAGE_TRANSFORM_TRANSLATION:
	set_transform_translation (registration, xf_out, xf_in, stage);
	break;
    case STAGE_TRANSFORM_VERSOR:
	set_transform_versor (registration, xf_out, xf_in, stage);
	break;
    case STAGE_TRANSFORM_QUATERNION:
	set_transform_quaternion (registration, xf_out, xf_in, stage);
	break;
    case STAGE_TRANSFORM_AFFINE:
	set_transform_affine (registration, xf_out, xf_in, stage);
	break;
    case STAGE_TRANSFORM_BSPLINE:
	set_transform_bspline (registration, xf_out, xf_in, stage);
	break;
    case STAGE_TRANSFORM_ALIGN_CENTER:
	set_transform_versor(registration, xf_out, xf_in, stage);
	break;
    }
    registration->SetInitialTransformParameters (
	registration->GetTransform()->GetParameters());

    if (stage->xform_type != STAGE_TRANSFORM_BSPLINE) {
	std::cout << "Intial Parameters = " 
		  << registration->GetTransform()->GetParameters() << std::endl;
    }
}

void
set_observer (RegistrationType::Pointer registration,
	      Stage_parms* stage)
{
    typedef Optimization_Observer OOType;
    OOType::Pointer observer = OOType::New();
    observer->Set_Stage_parms (registration, stage);
    registration->GetOptimizer()->AddObserver(itk::StartEvent(), observer);
    registration->GetOptimizer()->AddObserver(itk::InitializeEvent(), observer);
    registration->GetOptimizer()->AddObserver(itk::IterationEvent(), observer);
    registration->GetOptimizer()->AddObserver(itk::FunctionEvaluationIterationEvent(), observer);
    registration->GetOptimizer()->AddObserver(itk::ProgressEvent(), observer);
    registration->GetOptimizer()->AddObserver(itk::EndEvent(), observer);
}

void
set_xf_out (Xform *xf_out, 
	    RegistrationType::Pointer registration,
	    Stage_parms *stage)
{
    switch (stage->xform_type) {
    case STAGE_TRANSFORM_TRANSLATION:
	{
	    typedef TranslationTransformType * XfPtr;
	    XfPtr transform = static_cast<XfPtr>(registration->GetTransform());
	    xf_out->set_trn (transform);
	}
	break;
    case STAGE_TRANSFORM_VERSOR:
	{
	    typedef VersorTransformType * XfPtr;
	    XfPtr transform = static_cast<XfPtr>(registration->GetTransform());
	    xf_out->set_vrs (transform);
	}
	break;
    case STAGE_TRANSFORM_QUATERNION:
	{
	    typedef QuaternionTransformType * XfPtr;
	    XfPtr transform = static_cast<XfPtr>(registration->GetTransform());
	    xf_out->set_quat (transform);
	}
	break;
    case STAGE_TRANSFORM_AFFINE:
	{
	    typedef AffineTransformType * XfPtr;
	    XfPtr transform = static_cast<XfPtr>(registration->GetTransform());
	    xf_out->set_aff (transform);
	}
	break;
    case STAGE_TRANSFORM_BSPLINE:
	{
	    typedef BsplineTransformType * XfPtr;
	    XfPtr transform = static_cast<XfPtr>(registration->GetTransform());
	    xf_out->set_itk_bsp (transform);
	}
	break;
	case STAGE_TRANSFORM_ALIGN_CENTER:
	{
	    typedef VersorTransformType * XfPtr;
	    XfPtr transform = static_cast<XfPtr>(registration->GetTransform());
	    xf_out->set_vrs(transform);
	}
	break;
    }
}

void
do_itk_registration_stage (
    Registration_Data* regd, 
    Xform *xf_out, 
    Xform *xf_in, 
    Stage_parms* stage
)
{
    RegistrationType::Pointer registration = RegistrationType::New();

    /* Subsample fixed & moving images */
    FloatImageType::Pointer fixed_ss = subsample_image (
	regd->fixed_image->itk_float(), 
	stage->fixed_subsample_rate[0], 
	stage->fixed_subsample_rate[1], 
	stage->fixed_subsample_rate[2], 
	stage->background_val);
    FloatImageType::Pointer moving_ss = subsample_image (
	regd->moving_image->itk_float(), 
	stage->moving_subsample_rate[0], 
	stage->moving_subsample_rate[1], 
	stage->moving_subsample_rate[2], 
	stage->background_val);

    registration->SetFixedImage (fixed_ss);
    registration->SetMovingImage (moving_ss);

    set_metric (registration, stage);                    // must be after setting images
    set_mask_images (registration, regd, stage);         // must be after set_metric
    set_fixed_image_region (registration, regd, stage);  // must be after set_mask_images
    show_stats (registration);
    set_transform (registration, xf_out, xf_in, stage);  // must be after set_fixed_image_region
    set_optimization (registration, stage);

    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    registration->SetInterpolator (interpolator);
    set_observer (registration, stage);

    try {
	if (stage->optim_type != OPTIMIZATION_NO_REGISTRATION) {
	    std::cout << std::endl << "Starting Registration" << std::endl;
	    registration->Update ();
	    std::cout << std::endl << "Registration done." << std::endl;
	}
    }
    catch (itk::ExceptionObject & err) {
	std::cerr << "Exception caught in itk registration." << std::endl;
	std::cerr << err << std::endl;
	exit (-1);
    }

    set_xf_out (xf_out, registration, stage);
}

void
do_itk_center_stage (
    Registration_Data* regd, Xform *xf_out, Xform *xf_in, Stage_parms* stage)
{
    typedef itk::CenteredTransformInitializer < 
	VersorTransformType, FloatImageType, FloatImageType 
	> TransformInitializerType;
    RegistrationType::Pointer registration = RegistrationType::New();

    registration->SetFixedImage (regd->fixed_image->itk_float());
    registration->SetMovingImage (regd->moving_image->itk_float());

    VersorTransformType::Pointer trn = VersorTransformType::New();
    TransformInitializerType::Pointer initializer 
	= TransformInitializerType::New();
	
    initializer->SetTransform(trn);
    initializer->SetFixedImage(registration->GetFixedImage());
    initializer->SetMovingImage(registration->GetMovingImage());
    initializer->GeometryOn();

    std::cout << "Centering images" << std::endl;
    initializer->InitializeTransform();
    registration->SetTransform(trn);
    set_xf_out (xf_out, registration, stage);
}
