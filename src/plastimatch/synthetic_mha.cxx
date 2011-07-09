/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "itkImageRegionIteratorWithIndex.h"

#include "itk_image.h"
#include "math_util.h"
#include "rtds.h"
#include "rtss.h"
#include "synthetic_mha.h"

static void 
synth_gauss (
    float *intens, 
    unsigned char *label,
    const FloatPoint3DType& phys, 
    const Synthetic_mha_parms *parms
)
{
    float f = 0;
    for (int d = 0; d < 3; d++) {
	float f1 = phys[d] - parms->gauss_center[d];
	f1 = f1 / parms->gauss_std[d];
	f += f1 * f1;
    }
    f = exp (-0.5 * f);	    /* f \in (0,1] */
    

    *intens = (1 - f) * parms->background + f * parms->foreground;
    *label = (f > 0.2) ? 1 : 0;
}

static void 
synth_rect (
    float *intens, 
    unsigned char *label,
    const FloatPoint3DType& phys, 
    const Synthetic_mha_parms *parms
)
{
    if (phys[0] >= parms->rect_size[0] 
	&& phys[0] <= parms->rect_size[1] 
	&& phys[1] >= parms->rect_size[2] 
	&& phys[1] <= parms->rect_size[3] 
	&& phys[2] >= parms->rect_size[4] 
	&& phys[2] <= parms->rect_size[5])
    {
	*intens = parms->foreground;
	*label = 1;
    } else {
	*intens = parms->background;
	*label = 0;
    }
}

static void 
synth_sphere (
    float *intens, 
    unsigned char *label,
    const FloatPoint3DType& phys, 
    const Synthetic_mha_parms *parms
)
{
    float f = 0;
    for (int d = 0; d < 3; d++) {
	float f1 = phys[d] - parms->sphere_center[d];
	f1 = f1 / parms->sphere_radius[d];
	f += f1 * f1;
    }
    if (f > 1.0) {
	*intens = parms->background;
	*label = 0;
    } else {
	*intens = parms->foreground;
	*label = 1;
    }
}

static void 
synth_enclosed_rect (
    float *intens, 
    unsigned char *label,
    const FloatPoint3DType& phys, 
    const Synthetic_mha_parms *parms
)
{
    float f = 0.;
    float p[56]={ 
	5,5,5,    95,10,95, 1,
	5,90,5,   95,95,95, 1,
	5,5,5,    10,95,95, 1,
	90,5,5,   95,95,95, 1,
	5,5,5,    95,95,10, 1,
	5,5,90,   95,95,95, 1, 
	35,35,35, 70,70,70, parms->enclosed_intens_f1, 
	20,20,20, 80,25,30, parms->enclosed_intens_f2
    };

    // center is at 0, size is ~200
    // must specify dimension 200, origin -100, voxel spacing 1 in Slicer
    for (int i=0;i<7*8;i++) {
	if ((i%7)!=6) {
	    p[i]-=50;
	}
    }
    
    for (int i=0;i<8;i++)
    {
	
	if (i==6) { 
	    p[7*i+0]+=parms->enclosed_xlat1[0]; 
	    p[7*i+1]+=parms->enclosed_xlat1[1];
	    p[7*i+2]+=parms->enclosed_xlat1[2];
	    p[7*i+3]+=parms->enclosed_xlat1[0]; 
	    p[7*i+4]+=parms->enclosed_xlat1[1];
	    p[7*i+5]+=parms->enclosed_xlat1[2];
	}
	if (i==7) { 
	    p[7*i+0]+=parms->enclosed_xlat2[0];
	    p[7*i+1]+=parms->enclosed_xlat2[1];
	    p[7*i+2]+=parms->enclosed_xlat2[2];
	    p[7*i+3]+=parms->enclosed_xlat2[0];
	    p[7*i+4]+=parms->enclosed_xlat2[1];
	    p[7*i+5]+=parms->enclosed_xlat2[2];
	}
	if (p[7*i+0]<phys[0] && phys[0]<p[7*i+3] &&
	    p[7*i+1]<phys[1] && phys[1]<p[7*i+4] &&
	    p[7*i+2]<phys[2] && phys[2]<p[7*i+5]) 
	{ f = p[6+7*i]; } 
    }

    *intens = (1 - f) * parms->background + f * parms->foreground;
    *label = 0;
}

//box + ellipsoid inside
static void
synth_osd (
    float *intens, 
    unsigned char *label,
    const FloatPoint3DType& phys, 
    const Synthetic_mha_parms *parms
)
{
    float f;
    float p[36]={ 
	5,5,5,  195,8,195, 
	5,193,5, 195,195,195, 
	5,5,5,  8,195,195, 
	193,5,5, 195,195,195, 
	5,5,5,  195,195,8, 
	5,5,193, 195,195,195  
    };

    // must specify dimension 200, origin -100, voxel spacing 1 in Slicer
    for (int i=0;i<36;i++) {
	p[i]-=100;
    } 
		
    f = 0.; *label=0;
    for (int i=0;i<6;i++)
    {

	if (p[6*i+0]<phys[0] && phys[0]<p[6*i+3] &&
	    p[6*i+1]<phys[1] && phys[1]<p[6*i+4] &&
	    p[6*i+2]<phys[2] && phys[2]<p[6*i+5]) 
	{ f = 1.; *label=0; } 
    }

    // sphere
    if (parms->pattern_ss == PATTERN_SS_ONE) {
        float rs1= 25, rs2= 25, rs3=25;
	float xs=101, ys=101, zs=101;

	xs-=100; ys-=100; zs-=100; 

	float rr = (phys[2]-zs)*(phys[2]-zs)/(rs1*rs1)
	    +(phys[1]-ys)*(phys[1]-ys)/(rs2*rs2)
	    +(phys[0]-xs)*(phys[0]-xs)/(rs3*rs3);

	if ( rr < 2 ) { f = 1.; *label=1; }
    }

    // two spheres
    if (parms->pattern_ss == PATTERN_SS_TWO_APART) {
        float rs1= 25, rs2= 25, rs3=25;
	float xs=-40, ys=1, zs=1;

	float rr = (phys[2]-zs)*(phys[2]-zs)/(rs1*rs1)
	    +(phys[1]-ys)*(phys[1]-ys)/(rs2*rs2)
	    +(phys[0]-xs)*(phys[0]-xs)/(rs3*rs3);

	if ( rr < 2 ) { f = 1.; *label=1;}

	xs=40, ys=1, zs=1;

	rr = (phys[2]-zs)*(phys[2]-zs)/(rs1*rs1)
	    +(phys[1]-ys)*(phys[1]-ys)/(rs2*rs2)
	    +(phys[0]-xs)*(phys[0]-xs)/(rs3*rs3);
	
	if ( rr < 2 ) { f = 1.; *label=2;}
    }

    // two spheres partially overlapping and one aside
    if (parms->pattern_ss == PATTERN_SS_TWO_OVERLAP_PLUS_ONE) {
        float rs1= 25, rs2= 25, rs3=25;
	float xs=-20, ys=1, zs=1;

	float rr = (phys[2]-zs)*(phys[2]-zs)/(rs1*rs1)
	    +(phys[1]-ys)*(phys[1]-ys)/(rs2*rs2)
	    +(phys[0]-xs)*(phys[0]-xs)/(rs3*rs3);

	if ( rr < 2 ) { f = 1.; *label=1;}

	xs=15, ys=25, zs=1;

	rr = (phys[2]-zs)*(phys[2]-zs)/(rs1*rs1)
	    +(phys[1]-ys)*(phys[1]-ys)/(rs2*rs2)
	    +(phys[0]-xs)*(phys[0]-xs)/(rs3*rs3);
	
	if ( rr < 2 ) { f = 1.; *label=2;}

	xs=30, ys=-55, zs=1;
	rs1 = rs2 = rs3 = 10;

	rr = (phys[2]-zs)*(phys[2]-zs)/(rs1*rs1)
	    +(phys[1]-ys)*(phys[1]-ys)/(rs2*rs2)
	    +(phys[0]-xs)*(phys[0]-xs)/(rs3*rs3);

	if ( rr < 2 ) { f = 1.; *label=3;}

    }

    // two spheres partially overlapping, one within, and one aside
    if (parms->pattern_ss == PATTERN_SS_TWO_OVERLAP_PLUS_ONE_PLUS_EMBED) {
        float rs1= 25, rs2= 25, rs3=25;
	float xs=-30, ys=1, zs=1;

	float rr = (phys[2]-zs)*(phys[2]-zs)/(rs1*rs1)
	    +(phys[1]-ys)*(phys[1]-ys)/(rs2*rs2)
	    +(phys[0]-xs)*(phys[0]-xs)/(rs3*rs3);

	if ( rr < 2 ) { f = 1.; *label=1;}

	xs=5, ys=25, zs=1;

	rr = (phys[2]-zs)*(phys[2]-zs)/(rs1*rs1)
	    +(phys[1]-ys)*(phys[1]-ys)/(rs2*rs2)
	    +(phys[0]-xs)*(phys[0]-xs)/(rs3*rs3);
	
	if ( rr < 2 ) { f = 1.; *label=2;}

	xs=30, ys=-55, zs=1;
	rs1 = rs2 = rs3 = 10;

	rr = (phys[2]-zs)*(phys[2]-zs)/(rs1*rs1)
	    +(phys[1]-ys)*(phys[1]-ys)/(rs2*rs2)
	    +(phys[0]-xs)*(phys[0]-xs)/(rs3*rs3);

	if ( rr < 2 ) { f = 1.; *label=3;}

	xs=13, ys=33, zs=1;
	rs1 = rs2 = rs3 = 8;

	rr = (phys[2]-zs)*(phys[2]-zs)/(rs1*rs1)
	    +(phys[1]-ys)*(phys[1]-ys)/(rs2*rs2)
	    +(phys[0]-xs)*(phys[0]-xs)/(rs3*rs3);

	if ( rr < 2 ) { f = 0.5; *label=4;}

    }

    *intens = (1 - f) * parms->background + f * parms->foreground;
}

static void 
synth_donut (
    float *intens, 
    unsigned char *label,
    const FloatPoint3DType& phys, 
    const Synthetic_mha_parms *parms
)
{
    /* Set default values */
    *intens = parms->background;
    *label = 0;

    float p[3];
    for (int d = 0; d < 3; d++) {
	p[d] = (phys[d] - parms->donut_center[d]) / parms->donut_radius[d];
    }

    float dist = sqrt (p[0]*p[0] + p[1]*p[1]);

    /* Compute which ring we are inside */
    float ring_width = 1 / (float) parms->donut_rings;
    int ring_no = floor (dist / ring_width);

    /* If outside of all rings, return */
    if (ring_no >= parms->donut_rings) {
	return;
    }

    /* If within "background ring", return */
    if ((parms->donut_rings - ring_no) % 2 == 0) {
	return;
    }

    /* Compute distance from ring center */
    float ring_offset_1 = dist - ring_no * ring_width;
    float ring_offset_2 = (ring_no + 1) * ring_width - dist;
    float ring_offset = 0.5 * ring_width 
	- std::min (ring_offset_1, ring_offset_2);
    ring_offset = ring_offset / ring_width;

    /* If distance within donut, set to foreground */
    float dist_3d_sq = ring_offset * ring_offset + p[2] * p[2];

    if (dist_3d_sq < 1.) {
	*intens = parms->foreground;
	*label = 1;
    }
}


static float 
shifttanh (float x)
{
    return 0.5* ((exp(x)-exp(-x))/(exp(x)+exp(-x)) + 1);
}

//synthetic dose distribution
static float 
intens_dosemha (
    const FloatPoint3DType& phys, 
    float xlat1[3], 
    float xlat2[3],
    float f1, 
    float f2, 
    Pattern_structset_type pattern_ss
)
{
    float f=0;
    float x0=0, y0=0, z0=0;
    float sigma=30;

    if (pattern_ss == PATTERN_SS_ONE) { x0=1, y0=1, z0=1; }
    if (pattern_ss == PATTERN_SS_TWO_APART) { x0=-40, y0=1, z0=1; }
    if (pattern_ss == PATTERN_SS_TWO_OVERLAP_PLUS_ONE) { x0=-20, y0=1, z0=1; }
    if (pattern_ss == PATTERN_SS_TWO_OVERLAP_PLUS_ONE_PLUS_EMBED) { x0=30, y0=-55, z0=1, sigma=12; }

    float r = (phys[0]-x0)*(phys[0]-x0)+
	      (phys[1]-y0)*(phys[1]-y0)+
	      (phys[2]-z0)*(phys[2]-z0);

    f  += exp(-r/(sigma*sigma));

    /*
    r = (phys[0]-100.)*(phys[0]-100.)+
        (phys[1]-100.)*(phys[1]-100.)+
        (phys[2]-100.)*(phys[2]-100.)/(1.2*1.2);

    f += exp(-r/(30*30));
    */
    f = 20 * shifttanh(2.5*f-0.3);

    return f;
}

void
synthetic_mha (
    Rtds *rtds,
    Synthetic_mha_parms *parms
)
{
    /* Create ITK images for intensity, ss, and dose */
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
	float intens = 0.0f;
	unsigned char label_uchar = 0;

	/* Get 3D coordinates of voxel */
	FloatImageType::IndexType idx = it_out.GetIndex ();
	im_out->TransformIndexToPhysicalPoint (idx, phys);

	/* Compute intensity and label */
	switch (parms->pattern) {
	case PATTERN_GAUSS:
	    synth_gauss (&intens, &label_uchar, phys, parms);
	    break;
	case PATTERN_RECT:
	    synth_rect (&intens, &label_uchar, phys, parms);
	    break;
	case PATTERN_SPHERE:
	    synth_sphere (&intens, &label_uchar, phys, parms);
	    break;
	case PATTERN_ENCLOSED_RECT:
	    synth_enclosed_rect (&intens, &label_uchar, phys, parms);
	    break;
	case PATTERN_OBJSTRUCTDOSE:
	    synth_osd (&intens, &label_uchar, phys, parms);
	    break;
	case PATTERN_DONUT:
	    synth_donut (&intens, &label_uchar, phys, parms);
	    break;
	default:
	    intens = 0.0f;
	    label_uchar = 0;
	    break;
	}

	/* Set intensity */
	it_out.Set (intens);

	/* Set structure */
	if (parms->m_want_ss_img) {
	    uchar_img_it.Set (label_uchar); 
	    ++uchar_img_it;
	}

	/* Set dose */
	if (parms->m_want_dose_img) {
	    float dose = 0.;
	    if (parms->pattern == PATTERN_OBJSTRUCTDOSE) {
		dose = intens_dosemha (phys,
		    parms->enclosed_xlat1, parms->enclosed_xlat2,
		    parms->enclosed_intens_f1, parms->enclosed_intens_f2, 
		    parms->pattern_ss);
	    } else {
		const float thresh = parms->background + 
		    0.5 * (parms->foreground - parms->background);
		if (parms->foreground > parms->background 
		    && intens > thresh)
		{
		    dose = 15;
		} else if (parms->foreground < parms->background 
		    && intens < thresh)
		{
		    dose = 15;
		} else {
		    dose = 0;
		}
	    }
	    dose_img_it.Set (dose);
	    ++dose_img_it;
	}
    }

    /* Insert images into rtds */
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
