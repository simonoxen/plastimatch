/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "itkImageRegionIteratorWithIndex.h"

#include "itk_image.h"
#include "math_util.h"
#include "rtds.h"
#include "rtss.h"
#include "synthetic_mha.h"

float intens_enclosed(FloatPoint3DType phys, float f1, float f2)
{
    float f;
//	float f1, f2;
    float p[56]={ 5,5,5,  95,10,95, 1,
		  5,90,5, 95,95,95, 1,
		  5,5,5,  10,95,95, 1,
		  90,5,5, 95,95,95, 1,
		  5,5,5,  95,95,10, 1,
		  5,5,90, 95,95,95, 1, 
		  35,35,35, 70,70, 70, f1,
		  20,20,20, 80,25,30, f2
    };
	

    for(int i=0;i<7*8;i++) 
    { if ( (i%7)!=6 )
	    p[i]-=50; /*p[i]*=2;*/ } //center is at 0, size is ~200
    // must specify dimension 200, origin -100, voxel spacing 1 in Slicer
		
    f = 0.;
    for (int i=0;i<8;i++)
	if (p[7*i+0]<phys[0] && phys[0]<p[7*i+3] &&
	    p[7*i+1]<phys[1] && phys[1]<p[7*i+4] &&
	    p[7*i+2]<phys[2] && phys[2]<p[7*i+5]) 
	{ f = p[6+7*i]; } 
    return f;
}



void
synthetic_mha (
    Rtds *rtds,
    Synthetic_mha_parms *parms
)
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

    UCharImageType::Pointer uchar_img = UCharImageType::New();
    typedef itk::ImageRegionIteratorWithIndex< UCharImageType > 
	UCharIteratorType;
    UCharIteratorType uchar_img_it;
    if (parms->m_want_ss_img) {
	uchar_img->SetRegions(rg);
	uchar_img->SetOrigin(og);
	uchar_img->SetSpacing(sp);
	uchar_img->Allocate();
	uchar_img_it = UCharIteratorType (uchar_img, 
	    uchar_img->GetLargestPossibleRegion());
	uchar_img_it.GoToBegin();
    }

    FloatImageType::Pointer dose_img = FloatImageType::New();
    typedef itk::ImageRegionIteratorWithIndex< FloatImageType > 
	FloatIteratorType;
    FloatIteratorType dose_img_it;
    if (parms->m_want_dose_img) {
	dose_img->SetRegions(rg);
	dose_img->SetOrigin(og);
	dose_img->SetSpacing(sp);
	dose_img->Allocate();
	dose_img_it = FloatIteratorType (dose_img, 
	    dose_img->GetLargestPossibleRegion());
	dose_img_it.GoToBegin();
    }

    /* Iterate through image, setting values */
    typedef itk::ImageRegionIteratorWithIndex< FloatImageType > IteratorType;
    IteratorType it_out (im_out, im_out->GetLargestPossibleRegion());
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
	case PATTERN_ENCLOSED_RECT:
	    f = intens_enclosed(phys, parms->f1, parms->f2); // 0 to 1
	    f = (1 - f) * parms->background + f * parms->foreground;
	    break;
	default:
	    f = 0.0f;
	    break;
	}
	it_out.Set (f);

	if (parms->m_want_ss_img) {
	    const float thresh = parms->background + 
		0.5 * (parms->foreground - parms->background);
	    if (parms->foreground > parms->background && f > thresh) {
		uchar_img_it.Set (1);
	    } else if (parms->foreground < parms->background && f < thresh) {
		uchar_img_it.Set (1);
	    } else {
		uchar_img_it.Set (0);
	    }
	    ++uchar_img_it;
	}

	if (parms->m_want_dose_img) {
	    const float thresh = parms->background + 
		0.5 * (parms->foreground - parms->background);
	    if (parms->foreground > parms->background && f > thresh) {
		dose_img_it.Set (15);
	    } else if (parms->foreground < parms->background && f < thresh) {
		dose_img_it.Set (15);
	    } else {
		dose_img_it.Set (0);
	    }
	    ++dose_img_it;
	}
    }

    rtds->m_img = new Plm_image;
    rtds->m_img->set_itk (im_out);
    if (parms->m_want_ss_img) {
	rtds->m_ss_image = new Rtss (rtds);
	rtds->m_ss_image->m_ss_img = new Plm_image;
	rtds->m_ss_image->m_ss_img->set_itk (uchar_img);
    }
    if (parms->m_want_dose_img) {
	rtds->m_dose = new Plm_image;
	rtds->m_dose->set_itk (dose_img);
    }
}
