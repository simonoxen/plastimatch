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
#include "synthetic_mha.h"
#include "itk_image.h"
#include "getopt.h"

FloatImageType::Pointer
synthetic_mha (Synthetic_mha_parms* parms)
{

    /* Create ITK image */
    FloatImageType::SizeType sz;
    FloatImageType::IndexType st;
    FloatImageType::RegionType rg;
    FloatImageType::PointType og;
    FloatImageType::SpacingType sp;
    FloatImageType::DirectionType dc;
    for (int d1 = 0; d1 < 3; d1++) {
	st[d1] = 0;
	sz[d1] = parms->dim[d1];
	sp[d1] = parms->spacing[d1];
	og[d1] = parms->origin[d1];
    }
    rg.SetSize (sz);
    rg.SetIndex (st);

    FloatImageType::Pointer im_out = FloatImageType::New();
    im_out->SetRegions(rg);
    im_out->SetOrigin(og);
    im_out->SetSpacing(sp);
    im_out->Allocate();

    /* Iterate through image, setting values */
    typedef itk::ImageRegionIteratorWithIndex< FloatImageType > IteratorType;
    IteratorType it_out (im_out, im_out->GetRequestedRegion());
    for (it_out.GoToBegin(); !it_out.IsAtEnd(); ++it_out) {
	FloatPoint3DType phys;
	float f = 0.0f;

	FloatImageType::IndexType idx = it_out.GetIndex ();
	im_out->TransformIndexToPhysicalPoint (idx, phys);
	switch (parms->pattern) {
	case PATTERN_GAUSS:
	    f = 0;
	    for (int d = 0; d < 3; d++) {
		float f1 = phys[d] - parms->gauss_center[d];
		f1 = f1 / parms->gauss_std[d];
		f += f1 * f1;
	    }
	    f = exp (-0.5 * f);	    /* f \in (0,1] */
	    f = (1 - f) * parms->background + f * parms->foreground;
	    break;
	case PATTERN_RECT:
	    if (phys[0] >= parms->rect_size[0] 
		&& phys[0] <= parms->rect_size[1] 
		&& phys[1] >= parms->rect_size[2] 
		&& phys[1] <= parms->rect_size[3] 
		&& phys[2] >= parms->rect_size[4] 
		&& phys[2] <= parms->rect_size[5])
	    {
		f = parms->foreground;
	    } else {
		f = parms->background;
	    }
	    break;
	case PATTERN_SPHERE:
	    f = 0;
	    for (int d = 0; d < 3; d++) {
		float f1 = phys[d] - parms->sphere_center[d];
		f1 = f1 / parms->sphere_radius[d];
		f += f1 * f1;
	    }
	    if (f > 1.0) {
		f = parms->background;
	    } else {
		f = parms->foreground;
	    }
	    break;
	default:
	    f = 0.0f;
	    break;
	}
	it_out.Set (f);
    }
    return im_out;
}
