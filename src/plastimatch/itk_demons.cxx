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
#include "itkDemonsRegistrationFilter.h"
#include "itkLinearInterpolateImageFunction.h"

#include "itk_image.h"
#include "logfile.h"
#include "plm_parms.h"
#include "plm_timer.h"
#include "resample_mha.h"
#include "xform.h"

typedef itk::DemonsRegistrationFilter<
    FloatImageType,
    FloatImageType,
    DeformationFieldType> DemonsFilterType;

class Demons_Observer : public itk::Command
{
public:
    typedef Demons_Observer Self;
    typedef itk::Command Superclass;
    typedef itk::SmartPointer<Demons_Observer> Pointer;
    itkNewMacro (Demons_Observer);

public:
    Plm_timer timer;
    int m_feval;

protected:
    Demons_Observer() {
	plm_timer_start (&timer);
	m_feval = 0;
    };

public:
    void Execute(itk::Object *caller, const itk::EventObject & event)
    {
	Execute( (const itk::Object *)caller, event);
    }

    void Execute(const itk::Object * object, const itk::EventObject & event)
    {
	const DemonsFilterType * filter =
	    dynamic_cast< const DemonsFilterType* >(object);
	double val = filter->GetMetric();
	double duration = plm_timer_report (&timer);
	if (typeid(event) == typeid(itk::IterationEvent)) {
	    logfile_printf ("MSE [%4d] %9.3f [%6.3f secs]\n", 
		m_feval, val, duration);
	    plm_timer_start (&timer);
	    m_feval++;
	}
	else {
	    std::cout << "Unknown event type." << std::endl;
	    event.Print(std::cout);
	}
    }
};

static void
deformation_stats (DeformationFieldType::Pointer vf)
{
    typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
    FieldIterator fi (vf, vf->GetLargestPossibleRegion());
    const DeformationFieldType::SizeType vf_size 
	= vf->GetLargestPossibleRegion().GetSize();
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
do_demons_stage_internal (
    Registration_Data* regd, 
    Xform *xf_out, 
    Xform *xf_in,
    Stage_parms* stage)
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
		 Stage_parms* stage)
{
    do_demons_stage_internal (regd, xf_out, xf_in, stage);
    printf ("Deformation stats (out)\n");
    deformation_stats (xf_out->get_itk_vf());
}
