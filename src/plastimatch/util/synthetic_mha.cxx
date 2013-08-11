/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "itkImageRegionIteratorWithIndex.h"

#include "itk_directions.h"
#include "itk_image_type.h"
#include "itk_point.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_math.h"
#include "rt_study.h"
#include "rtss.h"
#include "segmentation.h"
#include "synthetic_mha.h"

static void 
synth_dose (
    float *intens, 
    unsigned char *label,
    const FloatPoint3DType& phys, 
    const Synthetic_mha_parms *parms
)
{
    /* sorry for this mess... */
    float x,x0,x1,y,y0,y1,f0,f1;

    float p[3];
    p[0] = phys[0] - parms->dose_center[0];
    p[1] = phys[1] - parms->dose_center[1];
    p[2] = phys[2] - parms->dose_center[2];

    /* uniform central dose */
    if (p[0] >= parms->dose_size[0] + parms->penumbra
        && p[0] <= parms->dose_size[1] - parms->penumbra
        && p[1] >= parms->dose_size[2] + parms->penumbra
        && p[1] <= parms->dose_size[3] - parms->penumbra
        && p[2] >= parms->dose_size[4] 
        && p[2] <= parms->dose_size[5])
    {
        *intens = parms->foreground;
        *label = 1;
    } else {
        *intens = parms->background;
        *label = 0;
    }

    if (p[2] >= parms->dose_size[4] && p[2] <= parms->dose_size[5]) {
        /* penumbra edges */
        if (p[1] > parms->dose_size[2]+parms->penumbra && p[1] < parms->dose_size[3]-parms->penumbra){
            x  = p[0];
            x0 = parms->dose_size[0];
            x1 = parms->dose_size[0] + parms->penumbra;
            f0 = parms->background;
            f1 = parms->foreground;
            if (x >= x0 && x < x1) {
                *intens = f0 + (x-x0)*((f1-f0)/(x1-x0));
            }

            x0 = parms->dose_size[1] - parms->penumbra;
            x1 = parms->dose_size[1];
            f0 = parms->foreground;
            f1 = parms->background;
            if (x >= x0 && x < x1) {
                *intens = f0 + (x-x0)*((f1-f0)/(x1-x0));
            }
        }
        if (p[0] > parms->dose_size[0]+parms->penumbra && p[0] < parms->dose_size[1]-parms->penumbra){
            y  = p[1];
            y0 = parms->dose_size[2];
            y1 = parms->dose_size[2] + parms->penumbra;
            f0 = parms->background;
            f1 = parms->foreground;
            if ((p[1] >= y0 && p[1] < y1)) {
                *intens = f0 + (y-y0)*((f1-f0)/(y1-y0));
            }

            y0 = parms->dose_size[3] - parms->penumbra;
            y1 = parms->dose_size[3];
            f0 = parms->foreground;
            f1 = parms->background;
            if (y >= y0 && y < y1) {
                *intens = f0 + (y-y0)*((f1-f0)/(y1-y0));
            }
        }
        
        /* penumbra corners */
        x = p[0];
        y = p[1];
        x0 = parms->dose_size[0];
        x1 = parms->dose_size[0] + parms->penumbra;
        y0 = parms->dose_size[2];
        y1 = parms->dose_size[2] + parms->penumbra;
        f0 = parms->background;
        f1 = parms->foreground;
        if (x > x0 && x < x1 && y > y0 && y < y1) {
            *intens = ((f0)/((x1-x0)*(y1-y0)))*(x1-x)*(y1-y) +
                ((f0)/((x1-x0)*(y1-y0)))*(x-x0)*(y1-y) +
                ((f0)/((x1-x0)*(y1-y0)))*(x1-x)*(y-y0) +
                ((f1)/((x1-x0)*(y1-y0)))*(x-x0)*(y-y0);
        }
        x = p[0];
        y = p[1];
        x0 = parms->dose_size[1] - parms->penumbra;
        x1 = parms->dose_size[1];
        y0 = parms->dose_size[2];
        y1 = parms->dose_size[2] + parms->penumbra;
        f0 = parms->background;
        f1 = parms->foreground;
        if (x > x0 && x < x1 && y > y0 && y < y1) {
            *intens = ((f0)/((x1-x0)*(y1-y0)))*(x1-x)*(y1-y) +
                ((f0)/((x1-x0)*(y1-y0)))*(x-x0)*(y1-y) +
                ((f1)/((x1-x0)*(y1-y0)))*(x1-x)*(y-y0) +
                ((f0)/((x1-x0)*(y1-y0)))*(x-x0)*(y-y0);
        }
        x = p[0];
        y = p[1];
        x0 = parms->dose_size[0];
        x1 = parms->dose_size[0] + parms->penumbra;
        y0 = parms->dose_size[3] - parms->penumbra;
        y1 = parms->dose_size[3];
        f0 = parms->background;
        f1 = parms->foreground;
        if (x > x0 && x < x1 && y > y0 && y < y1) {
            *intens = ((f0)/((x1-x0)*(y1-y0)))*(x1-x)*(y1-y) +
                ((f1)/((x1-x0)*(y1-y0)))*(x-x0)*(y1-y) +
                ((f0)/((x1-x0)*(y1-y0)))*(x1-x)*(y-y0) +
                ((f0)/((x1-x0)*(y1-y0)))*(x-x0)*(y-y0);
        }
        x = p[0];
        y = p[1];
        x0 = parms->dose_size[1] - parms->penumbra;
        x1 = parms->dose_size[1];
        y0 = parms->dose_size[3] - parms->penumbra;
        y1 = parms->dose_size[3];
        f0 = parms->background;
        f1 = parms->foreground;
        if (x > x0 && x < x1 && y > y0 && y < y1) {
            *intens = ((f1)/((x1-x0)*(y1-y0)))*(x1-x)*(y1-y) +
                ((f0)/((x1-x0)*(y1-y0)))*(x-x0)*(y1-y) +
                ((f0)/((x1-x0)*(y1-y0)))*(x1-x)*(y-y0) +
                ((f0)/((x1-x0)*(y1-y0)))*(x-x0)*(y-y0);
        }
    }
} /* z-direction */

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
    f = exp (-0.5 * f);            /* f \in (0,1] */

    *intens = (1 - f) * parms->background + f * parms->foreground;
    *label = (f > 0.2) ? 1 : 0;
}

static void 
synth_grid (
    float *intens, 
    unsigned char *label,
    const FloatPoint3DType& phys, 
    const Synthetic_mha_parms *parms
)
{
    int ijk[3];
    ijk[0] = (phys[0]/parms->spacing[0]) + (parms->dim[0]/2);
    ijk[1] = (phys[1]/parms->spacing[1]) + (parms->dim[1]/2);
    ijk[2] = (phys[2]/parms->spacing[2]) + (parms->dim[2]/2);

    if (
        ((ijk[0] % parms->grid_spacing[0] == 0) && (ijk[1] % parms->grid_spacing[1] == 0)) ||
        ((ijk[1] % parms->grid_spacing[1] == 0) && (ijk[2] % parms->grid_spacing[2] == 0)) ||
        ((ijk[2] % parms->grid_spacing[2] == 0) && (ijk[1] % parms->grid_spacing[1] == 0)) ||
        ((ijk[2] % parms->grid_spacing[2] == 0) && (ijk[0] % parms->grid_spacing[0] == 0))
    )
    {
        *intens = parms->foreground;
        *label = 1;
   } else {
        *intens = parms->background;
        *label = 0;
    }
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
        *intens 
            = (1 - parms->foreground_alpha) * (*intens) 
            + parms->foreground_alpha * parms->foreground;
        *label = 1;
    } else {
        *intens 
            = (1 - parms->background_alpha) * (*intens) 
            + parms->background_alpha * parms->background;
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
synth_multi_sphere (
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

static void 
synth_lung (
    float *intens, 
    unsigned char *label,
    const FloatPoint3DType& phys, 
    const Synthetic_mha_parms *parms
)
{
    /* Set default values */
    *intens = parms->background;
    *label = 0;

    /* Get distance from central axis */
    float d2 = sqrt (phys[0]*phys[0] + phys[1]*phys[1]);

    /* If outside chest wall, return */
    if (d2 > 150) {
        return;
    }

    /* If within chest wall, set chest wall intensity */
    if (d2 > 130) {
        *intens = parms->foreground;
        *label = 1;
        return;
    }

    /* Get distance from tumor */
    float p[3] = { 
        phys[0] - parms->lung_tumor_pos[0],
        phys[1] - parms->lung_tumor_pos[1],
        phys[2] - parms->lung_tumor_pos[2]
    };
    float d3 = sqrt (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);

    /* If within tumor, set tumor density */
    if (d3 < 20) {
        *label = 3;
        *intens = parms->foreground;
        return;
    }

    /* Otherwise, must be lung */
    *intens = -700;
    *label = 5;
}

static void 
synth_ramp (
    float *intens, 
    unsigned char *label,
    const FloatPoint3DType& phys, 
    const Synthetic_mha_parms *parms
)
{

    /* Get distance from origin */
    float d;
    if (parms->pattern == PATTERN_XRAMP) {
        d = phys[0] - parms->origin[0];
    }
    else if (parms->pattern == PATTERN_YRAMP) {
        d = phys[1] - parms->origin[1];
    }
    else {
        d = phys[2] - parms->origin[2];
    }

    /* Set intensity */
    *label = 0;
    *intens = d;
}

void
synthetic_mha (
    Rt_study *rtds,
    Synthetic_mha_parms *parms
)
{
    FloatImageType::SizeType sz;
    FloatImageType::IndexType st;
    FloatImageType::RegionType rg;
    FloatImageType::PointType og;
    FloatImageType::SpacingType sp;
    FloatImageType::DirectionType itk_dc;
    FloatImageType::Pointer im_out;

    if (parms->input_fn != "") {
        /* Input image was specified */
        Plm_image pi (parms->input_fn);
        im_out = pi.itk_float ();

        /* GCS FIX: Ideally, the calling code will set the alpha values 
           properly.  Instead, here we set the background alpha to 0, 
           with the understanding that the caller probably wants to 
           paste a rectangle onto the existing image */
        parms->background_alpha = 0.0f;

    } else {
        /* No input image specified */
        if (parms->fixed_fn.not_empty()) {
            /* Get geometry from fixed image */
            Plm_image pi (parms->fixed_fn);
            Plm_image_header pih;
            pih.set_from_plm_image (&pi);
            og = pih.m_origin;
            sp = pih.m_spacing;
            rg = pih.m_region;
            itk_dc = pih.m_direction;
            for (int d1 = 0; d1 < 3; d1++) {
                parms->origin[d1] = og[d1];
            }
        } else {
            /* Get geometry from command line parms */
            for (int d1 = 0; d1 < 3; d1++) {
                st[d1] = 0;
                sz[d1] = parms->dim[d1];
                sp[d1] = parms->spacing[d1];
                og[d1] = parms->origin[d1];
            }
            rg.SetSize (sz);
            rg.SetIndex (st);
            itk_direction_from_dc (&itk_dc, parms->dc);
        }

        /* Create new ITK image for intensity */
        im_out = FloatImageType::New();
        im_out->SetRegions (rg);
        im_out->SetOrigin (og);
        im_out->SetSpacing (sp);
        im_out->SetDirection (itk_dc);
        im_out->Allocate ();

        /* Initialize to background */
        im_out->FillBuffer (parms->background);
    }

    /* Create new ITK images for ss and dose */
    UCharImageType::Pointer ss_img = UCharImageType::New();
    typedef itk::ImageRegionIteratorWithIndex< UCharImageType > 
        UCharIteratorType;
    UCharIteratorType ss_img_it;

    if (parms->m_want_ss_img) {
        ss_img->SetRegions (rg);
        ss_img->SetOrigin (og);
        ss_img->SetSpacing (sp);
        ss_img->Allocate();
        ss_img_it = UCharIteratorType (ss_img, 
            ss_img->GetLargestPossibleRegion());
        ss_img_it.GoToBegin();
    }

    FloatImageType::Pointer dose_img = FloatImageType::New();
    typedef itk::ImageRegionIteratorWithIndex< FloatImageType > 
        FloatIteratorType;
    FloatIteratorType dose_img_it;
    if (parms->m_want_dose_img) {
        dose_img->SetRegions (rg);
        dose_img->SetOrigin (og);
        dose_img->SetSpacing (sp);
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
        //float intens = 0.0f;
        float intens = it_out.Get();
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
        case PATTERN_MULTI_SPHERE:
            synth_multi_sphere (&intens, &label_uchar, phys, parms);
            break;
        case PATTERN_DONUT:
            synth_donut (&intens, &label_uchar, phys, parms);
            break;
        case PATTERN_DOSE:
            synth_dose (&intens, &label_uchar, phys, parms);
            break;
        case PATTERN_GRID:
            synth_grid (&intens, &label_uchar, phys, parms);
            break;
        case PATTERN_LUNG:
            synth_lung (&intens, &label_uchar, phys, parms);
            break;
        case PATTERN_XRAMP:
        case PATTERN_YRAMP:
        case PATTERN_ZRAMP:
            synth_ramp (&intens, &label_uchar, phys, parms);
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
            ss_img_it.Set (label_uchar); 
            ++ss_img_it;
        }

        /* Set dose */
        if (parms->m_want_dose_img) {
            float dose = 0.;
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
            dose_img_it.Set (dose);
            ++dose_img_it;
        }
    }

    /* Insert images into rtds */
    Plm_image::Pointer pli = Plm_image::New();
    pli->set_itk (im_out);
    rtds->set_image (pli);
    if (parms->m_want_ss_img) {
        /* Create rtss & set into rtds */
        Segmentation::Pointer rtss = Segmentation::New (
            new Segmentation (rtds));
        rtds->set_rtss (rtss);

        /* Insert ss_image into rtss */
        rtss->set_ss_img (ss_img);

        /* Insert structure set into rtss */
        Rtss *rtss_ss = new Rtss;
        rtss->set_structure_set (rtss_ss);

        /* Add structure names */
        switch (parms->pattern) {
        case PATTERN_LUNG:
            rtss_ss->add_structure (Pstring ("Body"),
                Pstring(), 1, 0);
            rtss_ss->add_structure (Pstring ("Tumor"),
                Pstring(), 2, 1);
            rtss_ss->add_structure (Pstring ("Lung"),
                Pstring(), 3, 2);
            break;
        default:
            rtss_ss->add_structure (Pstring ("Foreground"),
                Pstring(), 1, 0);
        }
    }
    if (parms->m_want_dose_img) {
        rtds->set_dose (dose_img);
    }
}
