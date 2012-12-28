/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include "itkCenteredTransformInitializer.h"
#include "itkImageMaskSpatialObject.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkMutualInformationImageToImageMetric.h"
#include "itkRegularStepGradientDescentOptimizer.h"

#if defined(ITK_USE_OPTIMIZED_REGISTRATION_METHODS)
#include "itkOptMattesMutualInformationImageToImageMetric.h"
#include "itkOptMeanSquaresImageToImageMetric.h"
#else
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetric.h"
#endif

#include "compiler_warnings.h"
#include "itk_demons.h"
#include "itk_image_type.h"
#include "itk_optimizer.h"
#include "itk_registration.h"
#include "itk_registration_private.h"
#include "itk_resample.h"
#include "logfile.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "print_and_exit.h"
#include "registration_data.h"
#include "stage_parms.h"
#include "xform.h"

typedef itk::MeanSquaresImageToImageMetric <
    FloatImageType, FloatImageType > MSEMetricType;
typedef itk::MutualInformationImageToImageMetric <
    FloatImageType, FloatImageType > MIMetricType;
typedef itk::MattesMutualInformationImageToImageMetric <
    FloatImageType, FloatImageType > MattesMIMetricType;
typedef itk::ImageToImageMetric < 
    FloatImageType, FloatImageType > MetricType;

typedef itk::ImageMaskSpatialObject< 3 > Mask_SOType;

typedef itk::LinearInterpolateImageFunction <
    FloatImageType, double >InterpolatorType;

void
itk_align_center (
    Registration_data* regd, Xform *xf_out, Xform *xf_in, Stage_parms* stage);

Itk_registration_private::Itk_registration_private (
    Registration_data* regd, 
    Xform *xf_out, 
    Xform *xf_in, 
    Stage_parms* stage
)
{
    this->regd = regd;
    this->xf_in = xf_in;
    this->xf_out = xf_out;
    this->stage = stage;
    this->best_value = DBL_MAX;
    this->xf_best = new Xform (*xf_in);
}

Itk_registration_private::~Itk_registration_private ()
{
    delete this->xf_best;
}

/* ITK throws exceptions when e.g. evaluating metrics with overlap 
   of less that 25%.  Identify these so we can continue processing. */
static bool
itk_sample_failure (const itk::ExceptionObject& err)
{
    std::string err_string = err.GetDescription();
    const char *t = "Too many samples map outside moving image buffer";

    if (err_string.find (t) != std::string::npos) {
        return true;
    } else {
        return false;
    }
}

double
Itk_registration_private::evaluate_initial_transform ()
{
    double value = DBL_MAX;
    MetricType *metric = registration->GetMetric();
    try {
        value = metric->GetValue (
            registration->GetInitialTransformParameters());
    }
    catch (itk::ExceptionObject & err) {
        if (itk_sample_failure (err)) {
            lprintf ("ITK failed with too few samples.\n");
            return value;
        }
        lprintf ("Exception caught in evaluate_initial_transform()\n");
        std::stringstream ss;
        ss << err << "\n";
        lprintf (ss.str().c_str());
        exit (-1);
    }
    return value;
}

void
Itk_registration_private::set_best_xform ()
{
    switch (stage->xform_type) {
    case STAGE_TRANSFORM_TRANSLATION:
        xf_best->set_trn (
            registration->GetTransform()->GetParameters());
        break;
    case STAGE_TRANSFORM_VERSOR:
        xf_best->set_vrs (
            registration->GetTransform()->GetParameters());
        break;
    case STAGE_TRANSFORM_QUATERNION:
        xf_best->set_quat (
            registration->GetTransform()->GetParameters());
        break;
    case STAGE_TRANSFORM_AFFINE:
        xf_best->set_aff (
            registration->GetTransform()->GetParameters());
        break;
    case STAGE_TRANSFORM_BSPLINE: {
        /* GCS FIX: The B-spline method still gives the last xform, 
           not the best xform  */
#if defined (commentout)
        typedef BsplineTransformType * XfPtr;
        XfPtr transform = static_cast<XfPtr>(registration->GetTransform());
        xf_best->set_itk_bsp (transform);
#endif
    }
    break;
    default:
        print_and_exit ("Error: unknown case in set_best_xform()\n");
        break;
    }
}

void
Itk_registration_private::set_metric ()
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
        metric->SetNumberOfHistogramBins(stage->mi_histogram_bins_fixed);
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
Itk_registration_private::set_mask_images ()
{
    if (regd->fixed_mask) {
        Mask_SOType::Pointer mask_so = Mask_SOType::New();
        mask_so->SetImage(regd->fixed_mask->itk_uchar());
        mask_so->Update();
        registration->GetMetric()->SetFixedImageMask (mask_so);
    }
    if (regd->moving_mask) {
        Mask_SOType::Pointer mask_so = Mask_SOType::New();
        mask_so->SetImage(regd->moving_mask->itk_uchar());
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
    Registration_data* regd,
    Stage_parms* stage
)
{
    FloatImageType::RegionType valid_region;
    FloatImageType::RegionType::IndexType valid_index;
    FloatImageType::RegionType::SizeType valid_size;

    FloatImageType::ConstPointer fi = static_cast < 
        FloatImageType::ConstPointer > (registration->GetFixedImage());

    for (int d = 0; d < 3; d++) {
        float ori = regd->fixed_region_origin[d] 
            + regd->fixed_region.GetIndex()[d] * regd->fixed_region_spacing[d];
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
Itk_registration_private::set_fixed_image_region ()
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
        Mask_SOPointer so = (Mask_SOPointer) 
            registration->GetMetric()->GetFixedImageMask();

        typedef itk::ImageRegionConstIteratorWithIndex < UCharImageType 
            > IteratorType;
        UCharImageType::RegionType region 
            = registration->GetFixedImage()->GetLargestPossibleRegion();
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
                    UNUSED_VARIABLE (updated);
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
        typedef itk::ImageRegionConstIteratorWithIndex < FloatImageType
            > IteratorType;
        FloatImageType::RegionType region 
            = registration->GetFixedImage()->GetLargestPossibleRegion();
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
                    UNUSED_VARIABLE (updated);
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
        registration->SetFixedImageRegion (
            registration->GetFixedImage()->GetLargestPossibleRegion());
    }
}

template<class T>
void
show_image_stats (T image)
{
    typedef typename T::ObjectType Img;
    const typename Img::SizeType& sz = image->GetLargestPossibleRegion().GetSize();
    const typename Img::PointType& ori = image->GetOrigin();
    const typename Img::SpacingType& sp = image->GetSpacing();
    const typename Img::DirectionType& di = image->GetDirection();

    lprintf ("Origin = %g %g %g\n", ori[0], ori[1], ori[2]);
    lprintf ("Dim = %g %g %g\n", sp[0], sp[1], sp[2]);
    lprintf ("Spacing = %d %d %d\n", sz[0], sz[1], sz[2]);
    lprintf ("Direction Cosines =\n");
    for (unsigned int d1 = 0; d1 < 3; d1++) {
        for (unsigned int d2 = 0; d2 < 3; d2++) {
            lprintf (" %g", di[d1][d2]);
        }
        lprintf ("\n");
    }
}

void
Itk_registration_private::show_stats ()
{
    show_image_stats(static_cast < FloatImageType::ConstPointer > (
            registration->GetFixedImage()));
    show_image_stats(static_cast < FloatImageType::ConstPointer > (
            registration->GetMovingImage()));
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
}

void
Itk_registration_private::set_transform ()
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
        std::stringstream ss;
        ss << "Intial Parameters = " 
            << registration->GetTransform()->GetParameters() << "\n";
        lprintf (ss.str().c_str());
    }
}

void
Itk_registration_private::set_xf_out ()
{
    if (stage->xform_type == STAGE_TRANSFORM_BSPLINE) {
        /* Do nothing */
    } else {
        *xf_out = *xf_best;
    }
}

void
itk_registration_stage (
    Registration_data* regd, 
    Xform *xf_out, 
    Xform *xf_in, 
    Stage_parms* stage
)
{
    /* center_align is handled separately */
    if (stage->xform_type == STAGE_TRANSFORM_ALIGN_CENTER) {
        return itk_align_center (regd, xf_out, xf_in, stage);
    }

    Itk_registration_private irp (regd, xf_out, xf_in, stage);
    irp.registration = RegistrationType::New();

    /* Subsample fixed & moving images */
    FloatImageType::Pointer fixed_ss = subsample_image (
        regd->fixed_image->itk_float(), 
        stage->fixed_subsample_rate[0], 
        stage->fixed_subsample_rate[1], 
        stage->fixed_subsample_rate[2], 
        stage->default_value);
    FloatImageType::Pointer moving_ss = subsample_image (
        regd->moving_image->itk_float(), 
        stage->moving_subsample_rate[0], 
        stage->moving_subsample_rate[1], 
        stage->moving_subsample_rate[2], 
        stage->default_value);

    irp.registration->SetFixedImage (fixed_ss);
    irp.registration->SetMovingImage (moving_ss);

    irp.set_metric ();              // must be after setting images
    irp.set_mask_images ();         // must be after set_metric
    irp.set_fixed_image_region ();  // must be after set_mask_images
    irp.show_stats ();
    irp.set_transform ();           // must be after set_fixed_image_region
    irp.set_optimization ();

    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    irp.registration->SetInterpolator (interpolator);
    irp.set_observer ();

    try {
        if (stage->optim_type != OPTIMIZATION_NO_REGISTRATION) {
            lprintf ("Starting ITK registration\n");
            irp.registration->Update ();
            lprintf ("ITK registration complete\n");
        }
    }
    catch (itk::ExceptionObject & err) {
        if (itk_sample_failure (err)) {
            lprintf ("ITK failed with too few samples.\n");
        } else {
            lprintf ("Exception caught in itk registration.\n");
            std::stringstream ss;
            ss << err << "\n";
            lprintf (ss.str().c_str());
            exit (-1);
        }
    }

    irp.set_xf_out ();

    /* There is an ITK bug which deletes the internal memory of a 
       BSplineDeformableTransform when the RegistrationMethod
       is destructed.  This is a workaround for that bug. */
    if (irp.stage->xform_type == STAGE_TRANSFORM_BSPLINE) {
        xf_out->get_itk_bsp()->SetParametersByValue (
            xf_out->get_itk_bsp()->GetParameters ());
    }
}

void
itk_align_center (
    Registration_data* regd, Xform *xf_out, Xform *xf_in, Stage_parms* stage)
{
#if defined (commentout)
    typedef itk::CenteredTransformInitializer < 
        VersorTransformType, FloatImageType, FloatImageType 
        > TransformInitializerType;
    TransformInitializerType::Pointer initializer 
        = TransformInitializerType::New();
        
    VersorTransformType::Pointer trn = VersorTransformType::New();
    initializer->SetTransform (trn);
    initializer->SetFixedImage (regd->fixed_image->itk_float());
    initializer->SetMovingImage (regd->moving_image->itk_float());
    initializer->GeometryOn ();

    lprintf ("Centering images\n");
    initializer->InitializeTransform();

    xf_out->set_vrs (trn);
#endif
    float fixed_center[3];
    float moving_center[3];
    itk_volume_center (fixed_center, regd->fixed_image->itk_float());
    itk_volume_center (moving_center, regd->moving_image->itk_float());

    itk::Array<double> trn_parms (3);
    trn_parms[0] = moving_center[0] - fixed_center[0];
    trn_parms[1] = moving_center[1] - fixed_center[1];
    trn_parms[2] = moving_center[2] - fixed_center[2];
    xf_out->set_trn (trn_parms);
}

/* Greg's notes about itk problems (ITK 3.20.1)
   There are two ways to set the ITK bspline parameters:

   SetParameters()
   SetCoefficientImage()

   SetParameters()
   ---------------
   You pass in a (const ParametersType&), and then it stashes a pointer 
   to that in m_InputParametersPointer.  Finally it calls WrapAsImages(), 
   which maps m_CoefficientImage[] arrays into memory of 
   m_InputParametersPointer

   SetCoefficientImage()
   ---------------------
   You pass in a (ImagePointer[]), and these get copied into 
   m_CoefficientImage[] arrays.   m_InputParametersPointer gets reset 
   to zero.

   SetParametersByValue()
   ----------------------
   You pass in a (const ParametersType&), and then it allocates a new 
   memory to contain the array.  It initializes m_InputParametersPointer
   with the internal buffer.  Finally it calls WrapAsImages(), 
   which maps m_CoefficientImage[] arrays into memory of 
   m_InputParametersPointer

   xform_to_itk_bsp (xf_out, xf_in, &pih, stage->grid_spac);
   -----------------------
   Computes the grid parameters using bsp_grid_from_img_grid(), then 
   copies these into BSplineTransform using xform_itk_bsp_set_grid ().
   Note: SetGridRegion() will zero the region, but does not allocate it.
*/
