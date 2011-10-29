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

    /* Create ITK image */
    DeformationFieldType::SizeType sz;
    DeformationFieldType::IndexType st;
    DeformationFieldType::RegionType rg;
    DeformationFieldType::PointType og;
    DeformationFieldType::SpacingType sp;
    DeformationFieldType::DirectionType dc;
    for (int d1 = 0; d1 < 3; d1++) {
	st[d1] = 0;
	sz[d1] = parms->dim[d1];
	sp[d1] = parms->spacing[d1];
	og[d1] = parms->origin[d1];
    }
    rg.SetSize (sz);
    rg.SetIndex (st);

    DeformationFieldType::Pointer im_out = DeformationFieldType::New();
    im_out->SetRegions(rg);
    im_out->SetOrigin(og);
    im_out->SetSpacing(sp);
    im_out->Allocate();

    /* Iterate through vf, setting values */
    typedef itk::ImageRegionIteratorWithIndex< DeformationFieldType > 
	IteratorType;
    IteratorType it_out (im_out, im_out->GetRequestedRegion());

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
	im_out->TransformIndexToPhysicalPoint (idx, phys);
	switch (parms->pattern) {
	case Synthetic_vf_parms::PATTERN_ZERO:
	case Synthetic_vf_parms::PATTERN_TRANSLATION:
	default:
	    /* Do nothing */
	    break;
	case Synthetic_vf_parms::PATTERN_RADIAL:
	    /* Do something (when implemented) */
	    break;
	}
	it_out.Set (disp);
    }
    return im_out;
}
