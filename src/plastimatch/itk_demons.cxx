/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>   // Note: gcc prefers c++-style includes
#include "itkArray.h"
#include "itkCommand.h"
#include "itkHistogramMatchingImageFilter.h"
#include "itkIdentityTransform.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkDemonsRegistrationFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkWarpImageFilter.h"

#include "getopt.h"
#include "itk_image.h"
#include "plm_parms.h"
#include "resample_mha.h"
#include "xform.h"

typedef itk::DemonsRegistrationFilter<
                            FloatImageType,
                            FloatImageType,
                            DeformationFieldType> DemonsFilterType;

#if defined (GCS_REARRANGING_STUFF)

class Demons_Observer : public itk::Command
{
public:
    typedef  Demons_Observer   Self;
    typedef  itk::Command             Superclass;
    typedef  itk::SmartPointer<Demons_Observer>  Pointer;
    itkNewMacro( Demons_Observer );
protected:
    Demons_Observer() {};
public:
    void Execute(itk::Object *caller, const itk::EventObject & event)
      {
	Execute( (const itk::Object *)caller, event);
      }

    void Execute(const itk::Object * object, const itk::EventObject & event)
      {
	 const DemonsFilterType * filter =
	  dynamic_cast< const DemonsFilterType * >( object );
	if( typeid( event ) != typeid( itk::IterationEvent ) )
	  {
	  return;
	  }
	std::cout << filter->GetMetric() << std::endl;
      }
};

static void
deformation_stats (DeformationFieldType::Pointer vf)
{
    typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
    FieldIterator fi (vf, vf->GetLargestPossibleRegion());
    //DeformationFieldType::IndexType index;
    const DeformationFieldType::SizeType vf_size = vf->GetLargestPossibleRegion().GetSize();
    double max_sq_len = 0.0;
    double avg_sq_len = 0.0;

    for (fi.GoToBegin(); !fi.IsAtEnd(); ++fi) {
	//index = fi.GetIndex();
	const FloatVectorType& d = fi.Get();
	double sq_len = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
	if (sq_len > max_sq_len) {
	    max_sq_len = sq_len;
	}
	avg_sq_len += sq_len;
    }
    avg_sq_len /= (vf_size[0] * vf_size[1] * vf_size[2]);

    printf ("VF_MAX = %g   VF_AVG = %g\n", max_sq_len, avg_sq_len);
}

DeformationFieldType::Pointer 
check_resample_vector_field (DeformationFieldType::Pointer vf,
			     FloatImageType::Pointer image)
{
    printf ("Setting deformation field\n");
    const DeformationFieldType::SpacingType& vf_spacing = vf->GetSpacing();
    printf ("Deformation field spacing is: %g %g %g\n", 
	    vf_spacing[0], vf_spacing[1], vf_spacing[2]);
    const DeformationFieldType::SizeType vf_size = vf->GetLargestPossibleRegion().GetSize();
    printf ("VF Size is %d %d %d\n", vf_size[0], vf_size[1], vf_size[2]);

    const FloatImageType::SizeType& img_size = image->GetLargestPossibleRegion().GetSize();
    printf ("IM Size is %d %d %d\n", img_size[0], img_size[1], img_size[2]);
    const FloatImageType::SpacingType& img_spacing = image->GetSpacing();
    printf ("Deformation field spacing is: %g %g %g\n", 
	    img_spacing[0], img_spacing[1], img_spacing[2]);

    if (vf_size[0] != img_size[0] || vf_size[1] != img_size[1] || vf_size[2] != img_size[2]) {
	printf ("Deformation stats (pre)\n");
	deformation_stats (vf);

	vf = vector_resample_image (vf, image);

	printf ("Deformation stats (post)\n");
	deformation_stats (vf);
	const DeformationFieldType::SizeType vf_size = vf->GetLargestPossibleRegion().GetSize();
	printf ("NEW VF Size is %d %d %d\n", vf_size[0], vf_size[1], vf_size[2]);
	const FloatImageType::SizeType& img_size = image->GetLargestPossibleRegion().GetSize();
	printf ("IM Size is %d %d %d\n", img_size[0], img_size[1], img_size[2]);
    }
    return vf;
}

static DeformationFieldType::Pointer
run_demons_resolution (Registration_Parms* regp,
		       FloatImageType::Pointer fixed_image,
		       FloatImageType::Pointer moving_image,
		       int* resolution,
		       DeformationFieldType::Pointer vf)
{
    DemonsFilterType::Pointer filter = DemonsFilterType::New();

    /* Subsample fixed & moving images */
    /* Multires:
       1. Subsample the images
       2. Solve the deformation
       3. Supersample deformation
      */
    FloatImageType::Pointer fixed_ss
	    = subsample_image (fixed_image, resolution[0], resolution[1], resolution[2]);
    FloatImageType::Pointer moving_ss
	    = subsample_image (moving_image, resolution[0], resolution[1], resolution[2]);

    filter->SetFixedImage (fixed_ss);
    filter->SetMovingImage (moving_ss);

    Demons_Observer::Pointer observer = Demons_Observer::New();
    filter->AddObserver (itk::IterationEvent(), observer);

    //filter->SetNumberOfIterations (3);  // 25
    //filter->SetStandardDeviations(1.0);  // 6.0
    filter->SetNumberOfIterations (regp->max_its);
    filter->SetStandardDeviations (regp->demons_std);

    if (vf) {
	vf = check_resample_vector_field (vf, fixed_ss);
	filter->SetInitialDeformationField (vf);
    }

    if (regp->max_its <= 0) {
	print_and_exit ("Error demons iterations must be greater than 0\n");
    }
    if (regp->demons_std <= 0.0001) {
	print_and_exit ("Error demons std must be greater than 0\n");
    }

#if defined (commentout)
    DemonsFilterType::DemonsRegistrationFunctionType *drfp = 
	dynamic_cast<DemonsFilterType::DemonsRegistrationFunctionType *>
	  (filter->GetDifferenceFunction().GetPointer());
    drfp->SetUseMovingImageGradient(1);

    filter->SetIntensityDifferenceThreshold (33.210);
#endif

    printf ("Ready to start registration.\n");
    filter->Update();
    printf ("Done with registration.  Writing output...\n");

    return filter->GetOutput();
}

void
write_warped_output (Registration_Parms* regp,
		     FloatImageType::Pointer fixed_image,
		     FloatImageType::Pointer moving_image,
		     DeformationFieldType::Pointer vf)
{
    typedef itk::WarpImageFilter<
			  FloatImageType,
			  FloatImageType,
			  DeformationFieldType > WarperType;
    typedef itk::LinearInterpolateImageFunction<
				   FloatImageType,
				   double > InterpolatorType;
    WarperType::Pointer warper = WarperType::New();
    InterpolatorType::Pointer interpolator = InterpolatorType::New();

    warper->SetInput (moving_image);

    warper->SetInterpolator (interpolator);
    warper->SetOutputSpacing (fixed_image->GetSpacing());
    warper->SetOutputOrigin (fixed_image->GetOrigin());
    warper->SetDeformationField (vf);
//    warper->Update();

    FloatImageType::Pointer output_image = warper->GetOutput();
    save_short (output_image, regp->image_out_fn);
}

DeformationFieldType::Pointer
set_transform_demons (RegistrationType::Pointer registration,
		      Registration_Parms* regp)
{
    DeformationFieldType::Pointer vf_in = DeformationFieldType::New();

    if (*(regp->vf_in_fn)) {
        printf ("vf_in image=%s\n", regp->vf_in_fn);
	printf ("Loading input vector field...");
	fflush (stdout);
	vf_in = load_float_field (regp->vf_in_fn);
	printf ("done!\n");
	fflush (stdout);
    } else {
	if (regp->init_type == TRANSFORM_FROM_FILE) {
	    char buf[1024];
	    FILE* fp;
	    
	    fp = fopen (regp->xform_in_fn, "r");
	    if (!fp) {
		print_and_exit ("Error: xform_in file not found\n");
	    }
	    if (!fgets(buf,1024,fp)) {
		print_and_exit ("Error reading from xform_in file.\n");
	    }

	    if (strcmp(buf,"ObjectType = MGH_XFORM_AFFINE\n")==0) {
		int r, num_args;
		float f;

		/* 3 parms is translation, 6 parms is a versor, 
		12 parms is an affine */
		num_args = 0;
		while (r = fscanf (fp, "%f",&f)) {
		    regp->init[num_args++] = (double) f;
		    if (num_args == 12) break;
		    if (!fp) break;
		}
		fclose(fp);
	
		if (num_args != 12) {
		    print_and_exit ("Wrong number of parameters in xform_in file.\n");
		}
		regp->init_type = TRANSFORM_AFFINE;
	    }
	}

	switch (regp->init_type) {
	case TRANSFORM_NONE:
	    vf_in = 0;
	    break;
	case TRANSFORM_AFFINE:
	    {
		AffineTransformType::Pointer v = AffineTransformType::New();
		AffineTransformType::ParametersType at(12);
		for (int i=0; i<12; i++) {
		    at[i] = regp->init[i];
		}
		v->SetParameters(at);
		std::cout << "Initial affine parms = " << v << std::endl;

		vf_in->SetRegions (registration->GetFixedImage()->GetLargestPossibleRegion());
		vf_in->SetOrigin (registration->GetFixedImage()->GetOrigin());
		vf_in->SetSpacing (registration->GetFixedImage()->GetSpacing());
		vf_in->Allocate();

		typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
		FieldIterator fi (vf_in, registration->GetFixedImage()->GetLargestPossibleRegion());

		fi.GoToBegin();
	    
		AffineTransformType::InputPointType fixed_point;
		AffineTransformType::OutputPointType moving_point;
		DeformationFieldType::IndexType index;

		FloatVectorType displacement;

		while (!fi.IsAtEnd()) {
		    index = fi.GetIndex();
		    vf_in->TransformIndexToPhysicalPoint (index, fixed_point);
		    moving_point = v->TransformPoint (fixed_point);
		    for (int r = 0; r < Dimension; r++) {
			displacement[r] = moving_point[r] - fixed_point[r];
		    }
		    fi.Set (displacement);
		    ++fi;
		}
	    }
	    break;
	case TRANSFORM_TRANSLATION:
	case TRANSFORM_VERSOR:
	case TRANSFORM_FROM_FILE:
	default:
	    not_implemented();
	    break;
	}
    }
    return vf_in;
}

DeformationFieldType::Pointer
set_transform_demons_old (Registration_Parms* regp)
{
    DeformationFieldType::Pointer vf_in = DeformationFieldType::New();

    if (*(regp->vf_in_fn)) {
        printf ("vf_in image=%s\n", regp->vf_in_fn);
	printf ("Loading input vector field...");
	fflush (stdout);
	vf_in = load_float_field (regp->vf_in_fn);
	printf ("done!\n");
	fflush (stdout);
    } else {
	switch (regp->init_type) {
	case TRANSFORM_NONE:
	    vf_in = 0;
	    break;
	case TRANSFORM_AFFINE:
	    not_implemented();
	    break;
	case TRANSFORM_TRANSLATION:
	case TRANSFORM_VERSOR:
	case TRANSFORM_FROM_FILE:
	default:
	    not_implemented();
	    break;
	}
    }
    return vf_in;
}

void
do_demons_registration (Registration_Parms* regp)
{
    printf ("fixed image=%s\n", regp->fixed_fn);
    printf ("Loading fixed image...");
    FloatImageType::Pointer fixed_image = load_float (regp->fixed_fn);
    fflush (stdout);
    printf ("done!\n");

    printf ("moving image=%s\n", regp->moving_fn);
    printf ("Loading moving image...");
    fflush (stdout);
    FloatImageType::Pointer moving_image = load_float (regp->moving_fn);
    printf ("done!\n");

    RegistrationType::Pointer registration = RegistrationType::New();

    if (regp->histoeq!=0) {
        // ZW, 4/18/6, toy the idea of histogram matching the 2 image volumes 
	// before registration 
	typedef float InternalPixelType;
        typedef itk::Image< InternalPixelType, Dimension > InternalImageType;
	typedef itk::HistogramMatchingImageFilter<
	    InternalImageType,
    	InternalImageType > MatchingFilterType;
	MatchingFilterType::Pointer matcher = MatchingFilterType::New();
        matcher->SetInput( moving_image );
	matcher->SetReferenceImage( fixed_image );    
        /* We then select the number of bins to represent the histograms and the number 
	of points or quantile values where the histogram is to be matched. */
        matcher->SetNumberOfHistogramLevels( 1024 );
	matcher->SetNumberOfMatchPoints( 7 );
	/* Simple background extraction is done by thresholding at the mean intensity. */
	matcher->ThresholdAtMeanIntensityOn();
//	matcher->ThresholdAtMeanIntensityOff();

	registration->SetFixedImage (fixed_image);
//      registration->SetMovingImage (moving_image);
	// using the histogram matched image instead
        registration->SetMovingImage ( matcher->GetOutput() );
    } else {
	registration->SetFixedImage (fixed_image);
	registration->SetMovingImage (moving_image);
    }

    DeformationFieldType::Pointer vf_in = DeformationFieldType::New();
    vf_in = set_transform_demons (registration, regp);

    DeformationFieldType::Pointer vf = vf_in;
    for (int res = 0; res < regp->res_level; res++) {
	vf = run_demons_resolution (regp, fixed_image, moving_image, 
				    regp->resolutions[res], vf);
	printf ("Deformation stats (out)\n");
	deformation_stats (vf);
    }
    
    /* If the deformation field is not full resolution, we need to 
       upsample the deformation field */
    vf = check_resample_vector_field (vf, fixed_image);

    /* Save warped output */
    write_warped_output (regp, fixed_image, moving_image, vf);

    /* Save deformation field */
    save_image (vf, regp->vf_out_fn);
}

#if defined (commentout)
/* This is the old, single resolution version */
/* DON'T DELETE UNTIL THE OUTPUT CODE IS WORKING */
void
do_demons_registration_old (Registration_Parms* regp)
{
    DeformationFieldType::Pointer vf_in = DeformationFieldType::New();

    printf ("fixed image=%s\n", regp->fixed_fn);
    printf ("Loading fixed image...");

    FloatImageType::Pointer fixed_image = load_float (regp->fixed_fn);

    fflush (stdout);
    printf ("done!\n");
    printf ("moving image=%s\n", regp->moving_fn);
    printf ("Loading moving image...");
    fflush (stdout);

    FloatImageType::Pointer moving_image = load_float (regp->moving_fn);

    printf ("done!\n");

    if (*(regp->vf_in_fn)) {
        printf ("vf_in image=%s\n", regp->vf_in_fn);
	printf ("Loading input vector field...");
	fflush (stdout);
	vf_in = load_float_field (regp->vf_in_fn);
	printf ("done!\n");
	fflush (stdout);
    }

    DemonsFilterType::Pointer filter = DemonsFilterType::New();

    Demons_Observer::Pointer observer = Demons_Observer::New();
    filter->AddObserver (itk::IterationEvent(), observer);

    filter->SetFixedImage (fixed_image);
    filter->SetMovingImage (moving_image);
    //filter->SetNumberOfIterations (3);  // 25
    //filter->SetStandardDeviations(1.0);  // 6.0
    filter->SetNumberOfIterations (regp->max_its);
    filter->SetStandardDeviations (regp->demons_std);
    if (*(regp->vf_in_fn)) {
	filter->SetInitialDeformationField (vf_in);
    }

    if (regp->max_its <= 0) {
	print_and_exit ("Error demons iterations must be greater than 0\n");
    }
    if (regp->demons_std <= 0.0001) {
	print_and_exit ("Error demons std must be greater than 0\n");
    }

    // Heh heh.  It's worth a try isn't it?
    //filter->GetMetric()->SetFixedImageMask(fixed_image);
    // Hmm, get metric seems to be getting the metric value
    //double e = filter->GetMetric();

    filter->Update();
    printf ("Done with registration.  Writing output...\n");

    typedef itk::WarpImageFilter<
			  FloatImageType,
			  FloatImageType,
			  DeformationFieldType  >     WarperType;
    typedef itk::LinearInterpolateImageFunction<
				   FloatImageType,
				   double          >  InterpolatorType;
    WarperType::Pointer warper = WarperType::New();
    InterpolatorType::Pointer interpolator = InterpolatorType::New();

/* #define OUTPUT_WARPED_PET 1 */

#if defined (OUTPUT_WARPED_PET)
    /* For displaying petpet */
    MhaReaderType::Pointer pet_rdr = MhaReaderType::New();
    load_mha_rdr(pet_rdr, "rigid_ox111_with_mask.mha");
    InputImageType::Pointer pet_input_image = pet_rdr->GetOutput();
    printf ("Calling Update() on pet image...\n");
    pet_rdr->Update();
    MovingCastFilterType::Pointer pet_caster = MovingCastFilterType::New();
    pet_caster->SetInput (pet_input_image);
    pet_caster->Update();
    FloatImageType::Pointer pet_image = pet_caster->GetOutput();
    printf ("Converting pet image to float...\n");
    pet_image->Update();

    warper->SetInput (pet_image);
#else
    /* For displaying petct */
    warper->SetInput (moving_image);
#endif
    warper->SetInterpolator (interpolator);
    warper->SetOutputSpacing (fixed_image->GetSpacing());
    warper->SetOutputOrigin (fixed_image->GetOrigin());
    warper->SetDeformationField (filter->GetOutput());

    typedef short OutputPixelType;
    typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
    typedef itk::CastImageFilter<
			FloatImageType,
			OutputImageType > CastFilterType;
    CastFilterType::Pointer  caster =  CastFilterType::New();
    caster->SetInput (warper->GetOutput());

    typedef itk::ImageFileWriter< OutputImageType >  WriterType;
    WriterType::Pointer      writer =  WriterType::New();
#if defined (OUTPUT_WARPED_PET)
    writer->SetFileName ("demons_ox221.mha");
#else
    writer->SetFileName (regp->image_out_fn);
#endif
    writer->SetInput (caster->GetOutput());
    writer->Update();

    /* Output the deformation field */
    printf ("Writing deformation field\n");
    typedef itk::ImageFileWriter< DeformationFieldType >  FieldWriterType;
    FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
    fieldWriter->SetInput (filter->GetOutput());
    fieldWriter->SetFileName (regp->vf_out_fn);
    try {
	fieldWriter->Update();
    }
    catch (itk::ExceptionObject& excp) {
	std::cerr << "Exception thrown " << std::endl;
	std::cerr << excp << std::endl;
    }

}

#endif

#endif /* GCS_REARRANGING_STUFF */

class Demons_Observer : public itk::Command
{
public:
    typedef  Demons_Observer   Self;
    typedef  itk::Command             Superclass;
    typedef  itk::SmartPointer<Demons_Observer>  Pointer;
    itkNewMacro( Demons_Observer );
protected:
    Demons_Observer() {};
public:
    void Execute(itk::Object *caller, const itk::EventObject & event)
      {
	Execute( (const itk::Object *)caller, event);
      }

    void Execute(const itk::Object * object, const itk::EventObject & event)
      {
	 const DemonsFilterType * filter =
	  dynamic_cast< const DemonsFilterType * >( object );
	if( typeid( event ) != typeid( itk::IterationEvent ) )
	  {
	  return;
	  }
	std::cout << filter->GetMetric() << std::endl;
      }
};

static void
deformation_stats (DeformationFieldType::Pointer vf)
{
    typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
    FieldIterator fi (vf, vf->GetLargestPossibleRegion());
    //DeformationFieldType::IndexType index;
    const DeformationFieldType::SizeType vf_size = vf->GetLargestPossibleRegion().GetSize();
    double max_sq_len = 0.0;
    double avg_sq_len = 0.0;

    for (fi.GoToBegin(); !fi.IsAtEnd(); ++fi) {
	//index = fi.GetIndex();
	const FloatVector3DType& d = fi.Get();
	double sq_len = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
	if (sq_len > max_sq_len) {
	    max_sq_len = sq_len;
	}
	avg_sq_len += sq_len;
    }
    avg_sq_len /= (vf_size[0] * vf_size[1] * vf_size[2]);

    printf ("VF_MAX = %g   VF_AVG = %g\n", max_sq_len, avg_sq_len);
}

static void
do_demons_stage_internal (Registration_Data* regd, 
			 Xform *xf_out, 
			 Xform *xf_in,
			 Stage_Parms* stage)
{
    DemonsFilterType::Pointer filter = DemonsFilterType::New();
    DeformationFieldType::Pointer vf;

    /* Subsample fixed & moving images */
    FloatImageType::Pointer fixed_ss
	    = subsample_image (regd->fixed_image->itk_float(), 
			       stage->fixed_subsample_rate[0], 
			       stage->fixed_subsample_rate[1], 
			       stage->fixed_subsample_rate[2], 
			       stage->background_val);
    FloatImageType::Pointer moving_ss
	    = subsample_image (regd->moving_image->itk_float(), 
			       stage->moving_subsample_rate[0], 
			       stage->moving_subsample_rate[1], 
			       stage->moving_subsample_rate[2], 
			       stage->background_val);

    filter->SetFixedImage (fixed_ss);
    filter->SetMovingImage (moving_ss);

    Demons_Observer::Pointer observer = Demons_Observer::New();
    filter->AddObserver (itk::IterationEvent(), observer);

    filter->SetNumberOfIterations (stage->max_its);
    filter->SetStandardDeviations (stage->demons_std);

    /* Get vector field of matching resolution */
    if (xf_in->m_type != STAGE_TRANSFORM_NONE) {
	xform_to_itk_vf (xf_out, xf_in, fixed_ss);
	filter->SetInitialDeformationField (xf_out->get_itk_vf());
    }

    if (stage->max_its <= 0) {
	print_and_exit ("Error demons iterations must be greater than 0\n");
    }
    if (stage->demons_std <= 0.0001) {
	print_and_exit ("Error demons std must be greater than 0\n");
    }

#if defined (commentout)
    DemonsFilterType::DemonsRegistrationFunctionType *drfp = 
	dynamic_cast<DemonsFilterType::DemonsRegistrationFunctionType *>
	  (filter->GetDifferenceFunction().GetPointer());
    drfp->SetUseMovingImageGradient(1);

    filter->SetIntensityDifferenceThreshold (33.210);
#endif

    printf ("Ready to start registration.\n");
    filter->Update();
    printf ("Done with registration.  Writing output...\n");

    xf_out->set_itk_vf (filter->GetOutput());
}

void
do_demons_stage (Registration_Data* regd, 
		 Xform *xf_out, 
		 Xform *xf_in,
		 Stage_Parms* stage)
{
    do_demons_stage_internal (regd, xf_out, xf_in, stage);
    printf ("Deformation stats (out)\n");
    deformation_stats (xf_out->get_itk_vf());
}
