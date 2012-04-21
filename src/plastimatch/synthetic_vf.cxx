/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* Correct mha files which have incorrect patient orientations */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "plm_config.h"
#include "math_util.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "synthetic_vf.h"
#include "itk_image.h"
#include "getopt.h"

DeformationFieldType::Pointer
synthetic_vf (Synthetic_vf_parms* parms)
{
    /* Create ITK vf */
    DeformationFieldType::Pointer vf_out = DeformationFieldType::New();
    printf ("Setting header\n");
    parms->pih.print ();
    itk_image_set_header (vf_out, &parms->pih);
    printf ("Header was set\n");
    vf_out->Allocate();

    /* Iterate through vf, setting values */
    typedef itk::ImageRegionIteratorWithIndex< DeformationFieldType > 
	IteratorType;
    IteratorType it_out (vf_out, vf_out->GetRequestedRegion());

    /* Stock displacements can be initialized outside of loop */
    FloatVector3DType disp;
    for (int d = 0; d < 3; d++) {
	switch (parms->pattern) {
	case Synthetic_vf_parms::PATTERN_ZERO:
	default:
	    disp[d] = 0;
	    break;
	case Synthetic_vf_parms::PATTERN_TRANSLATION:
	    disp[d] = parms->translation[d];
	    break;
	}
    }

    for (it_out.GoToBegin(); !it_out.IsAtEnd(); ++it_out) {
	FloatPoint3DType phys;

	DeformationFieldType::IndexType idx = it_out.GetIndex ();
	vf_out->TransformIndexToPhysicalPoint (idx, phys);
	switch (parms->pattern) {
	case Synthetic_vf_parms::PATTERN_ZERO:
	case Synthetic_vf_parms::PATTERN_TRANSLATION:
	default:
            /* Don't change disp */
	    break;
	case Synthetic_vf_parms::PATTERN_RADIAL:
	    /* Do something (when implemented) */
	    break;
	}
	it_out.Set (disp);
    }
    return vf_out;
}
