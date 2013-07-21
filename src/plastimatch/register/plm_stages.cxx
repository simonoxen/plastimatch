/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <stdlib.h>
#include <string.h>
#include "itkImageRegionConstIteratorWithIndex.h"

#include "bspline_xform.h"
#include "gpuit_bspline.h"
#include "gpuit_demons.h"
#include "itk_demons.h"
#include "itk_image_save.h"
#include "itk_registration.h"
#include "logfile.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_stages.h"
#include "plm_timer.h"
#include "plm_warp.h"
#include "registration_data.h"
#include "registration_parms.h"
#include "stage_parms.h"
#include "volume.h"
#include "xform.h"

#define FIXME_BACKGROUND_MAX (-1200)


/* This helps speed up the registration, by setting the bounding box to the 
   smallest size needed.  To find the bounding box, either use the extent 
   of the fixed_mask (if one is used), or by eliminating excess air 
   by thresholding */
static void
set_fixed_image_region_global (Registration_data* regd)
{
    int use_magic_value = 1;

    regd->fixed_region_origin = regd->fixed_image->itk_float()->GetOrigin();
    regd->fixed_region_spacing = regd->fixed_image->itk_float()->GetSpacing();

    if (regd->fixed_mask) {
        FloatImageType::RegionType::IndexType valid_index;
        FloatImageType::RegionType::SizeType valid_size;

        /* Search for bounding box of fixed mask */
        typedef itk::ImageRegionConstIteratorWithIndex< 
            UCharImageType > IteratorType;
        UCharImageType::RegionType region 
            = regd->fixed_mask->itk_uchar()->GetLargestPossibleRegion();
        IteratorType it (regd->fixed_mask->itk_uchar(), region);

        int first = 1;
        valid_index[0] = 0;
        valid_index[1] = 0;
        valid_index[2] = 0;
        valid_size[0] = 0;
        valid_size[1] = 0;
        valid_size[2] = 0;

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
                    for (int i = 0; i < 3; i++) {
                        if (valid_index[i] > idx[i]) {
                            valid_size[i] += valid_index[i] - idx[i];
                            valid_index[i] = idx[i];
                        }
                        if (idx[i] - valid_index[i] >= (long) valid_size[i]) {
                            valid_size[i] = idx[i] - valid_index[i] + 1;
                        }
                    }
                }
            }
        }
        regd->fixed_region.SetIndex(valid_index);
        regd->fixed_region.SetSize(valid_size);
    } else if (use_magic_value) {
        FloatImageType::RegionType::IndexType valid_index;
        FloatImageType::RegionType::SizeType valid_size;
        valid_index[0] = 0;
        valid_index[1] = 0;
        valid_index[2] = 0;
        valid_size[0] = 1;
        valid_size[1] = 1;
        valid_size[2] = 1;

        /* Make sure the image is ITK float */
        FloatImageType::Pointer fixed_image = regd->fixed_image->itk_float();

        /* Search for bounding box of patient */
        typedef itk::ImageRegionConstIteratorWithIndex <
            FloatImageType > IteratorType;
        FloatImageType::RegionType region 
            = fixed_image->GetLargestPossibleRegion();
        IteratorType it (fixed_image, region);

        int first = 1;
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            float c = it.Get();
            if (c > FIXME_BACKGROUND_MAX) {
                FloatImageType::RegionType::IndexType idx = it.GetIndex();
                if (first) {
                    first = 0;
                    valid_index = idx;
                    valid_size[0] = 1;
                    valid_size[1] = 1;
                    valid_size[2] = 1;
                } else {
                    for (int i = 0; i < 3; i++) {
                        if (valid_index[i] > idx[i]) {
                            valid_size[i] += valid_index[i] - idx[i];
                            valid_index[i] = idx[i];
                        }
                        if (idx[i] - valid_index[i] >= (long) valid_size[i]) {
                            valid_size[i] = idx[i] - valid_index[i] + 1;
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
            if (valid_size[i] + valid_index[i] 
                < fixed_image->GetLargestPossibleRegion().GetSize()[i])
            {
                valid_size[i]++;
            }
        }
        regd->fixed_region.SetIndex(valid_index);
        regd->fixed_region.SetSize(valid_size);
    } else {
        regd->fixed_region 
            = regd->fixed_image->itk_float()->GetLargestPossibleRegion();
    }
}

static void
save_output (
    Registration_data* regd, 
    Xform *xf_out, 
    const std::list<std::string>& xf_out_fn, 
    bool xf_out_itk,
    int img_out_fmt,
    Plm_image_type img_out_type,
    float default_value, 
    const char *img_out_fn,
    const char *vf_out_fn
)
{
    /* Save xf to all filenames in list */
    std::list<std::string>::const_iterator it;
    for (it = xf_out_fn.begin(); it != xf_out_fn.end(); ++it) {
        logfile_printf ("Writing transformation ...\n");
        if (xf_out_itk && xf_out->m_type == XFORM_GPUIT_BSPLINE) {
            Xform xf_tmp;
            Plm_image_header pih;
            pih.set_from_plm_image (regd->fixed_image);
            xform_to_itk_bsp (&xf_tmp, xf_out, &pih, 0);
            xform_save (&xf_tmp, (*it).c_str());
        } else {
            xform_save (xf_out, (*it).c_str());
        }
    }

    if (img_out_fn[0] || vf_out_fn[0]) {
        DeformationFieldType::Pointer vf;
        DeformationFieldType::Pointer *vfp;
        Plm_image im_warped;
        Plm_image *imp;
        Plm_image_header pih;

        if (vf_out_fn[0]) {
            vfp = &vf;
        } else {
            vfp = 0;
        }
        if (img_out_fn[0]) {
            imp = &im_warped;
        } else {
            imp = 0;
        }
        
        pih.set_from_plm_image (regd->fixed_image);

        logfile_printf ("Warping...\n");
        plm_warp (imp, vfp, xf_out, &pih, regd->moving_image, 
            default_value, 0, 1);

        if (img_out_fn[0]) {
            logfile_printf ("Saving image...\n");
            if (img_out_fmt == IMG_OUT_FMT_AUTO) {
                if (img_out_type == PLM_IMG_TYPE_UNDEFINED) {
                    im_warped.save_image (img_out_fn);
                } else {
                    im_warped.convert_and_save (img_out_fn, img_out_type);
                }
            } else {
                im_warped.save_short_dicom (img_out_fn, 0);
            }
        }
        if (vf_out_fn[0]) {
            logfile_printf ("Saving vf...\n");
            itk_image_save (vf, vf_out_fn);
        }
    }
}

static void
do_registration_stage (
    Registration_parms* regp, 
    Registration_data* regd, 
    Xform *xf_out, Xform *xf_in, 
    Stage_parms* stage)
{
    lprintf ("[1] xf_in->m_type = %d, xf_out->m_type = %d\n", 
        xf_in->m_type, xf_out->m_type);

    /* Run registration */
    if (stage->optim_type == OPTIMIZATION_DEMONS) {
        if (stage->impl_type == IMPLEMENTATION_ITK) {
            do_demons_stage (regd, xf_out, xf_in, stage);
        } else {
            do_gpuit_demons_stage (regd, xf_out, xf_in, stage);
        }
    }
    else if (stage->xform_type == STAGE_TRANSFORM_BSPLINE) {
        if (stage->impl_type == IMPLEMENTATION_ITK) {
            itk_registration_stage (regd, xf_out, xf_in, stage);
        } else {
            do_gpuit_bspline_stage (regp, regd, xf_out, xf_in, stage);
        }
    }
    else if (stage->xform_type == STAGE_TRANSFORM_ALIGN_CENTER) {
        itk_registration_stage (regd, xf_out, xf_in, stage);
        lprintf ("Centering done\n");
    }
    else {
        itk_registration_stage (regd, xf_out, xf_in, stage);
    }

    lprintf ("[2] xf_out->m_type = %d, xf_in->m_type = %d\n", 
        xf_out->m_type, xf_in->m_type);

    /* Save intermediate output */
    save_output (regd, xf_out, stage->xf_out_fn, stage->xf_out_itk, 
        stage->img_out_fmt, stage->img_out_type, 
        stage->default_value, stage->img_out_fn, stage->vf_out_fn);
}

static void
set_auto_subsampling (int subsample_rate[], Plm_image *pli)
{
    Plm_image_header pih (pli);
    
    for (int d = 0; d < 3; d++) {
        subsample_rate[d] = (pih.Size(d)+99) / 100;
    }
}

static void
set_automatic_parameters (Registration_data* regd, Registration_parms* regp)
{
    std::list<Stage_parms*>& stages = regp->get_stages();
    std::list<Stage_parms*>::iterator it;
    for (it = stages.begin(); it != stages.end(); it++) {
        Stage_parms* sp = *it;
        if (sp->subsampling_type == SUBSAMPLING_AUTO) {
            set_auto_subsampling (
                sp->fixed_subsample_rate, regd->fixed_image);
            set_auto_subsampling (
                sp->moving_subsample_rate, regd->moving_image);
        }
    }
}

static void
check_output_resolution (Xform* xf_out, Registration_data* regd)
{
    Volume *fixed = regd->fixed_image->get_vol ();
    int ss[3];
    Plm_image_header pih;
    float grid_spacing[3];

    if (xf_out->m_type != XFORM_GPUIT_BSPLINE) {
        return;
    }

    Bspline_xform *bxf_out = xf_out->get_gpuit_bsp();
    if ( (bxf_out->img_dim[0] != fixed->dim[0]) ||
         (bxf_out->img_dim[1] != fixed->dim[1]) ||
         (bxf_out->img_dim[2] != fixed->dim[2]) ) {

        ss[0] = fixed->dim[0] / bxf_out->img_dim[0];
        ss[1] = fixed->dim[1] / bxf_out->img_dim[1];
        ss[2] = fixed->dim[2] / bxf_out->img_dim[2];

        /* last stage was not [1 1 1], un-subsample the final xform */
        logfile_printf ("RESTORE NATIVE RESOLUTION: (%d %d %d), (1 1 1)\n",
                ss[0], ss[1], ss[2]);

        /* Transform input xform to gpuit vector field */
        pih.set_from_gpuit (
            fixed->dim, 
            fixed->offset,
            fixed->spacing, 
            fixed->direction_cosines
        );
        xf_out->get_grid_spacing (grid_spacing);
        xform_to_gpuit_bsp (xf_out, xf_out, &pih, grid_spacing);
    }
}

/* xf_result must be freed by caller */
void
do_registration_pure (
    Xform** xf_result,
    Registration_data* regd,
    Registration_parms* regp
)
{
    Xform *xf1 = new Xform;
    Xform *xf2 = new Xform;
    Xform *xf_in, *xf_out, *xf_tmp;

    xf_in = xf1;
    xf_out = xf2;

    /* Load initial guess of xform */
    if (regp->xf_in_fn[0]) {
        xform_load (xf_out, regp->xf_in_fn);
    }

    /* Set fixed image region */
    set_fixed_image_region_global (regd);

    /* Set automatic parameters based on image size */
    set_automatic_parameters (regd, regp);

    std::list<Stage_parms*>& stages = regp->get_stages();
    std::list<Stage_parms*>::iterator it;
    for (it = stages.begin(); it != stages.end(); it++) {
        /* Swap xf_in and xf_out */
        xf_tmp = xf_out; xf_out = xf_in; xf_in = xf_tmp;
        /* Run registation, results are stored in xf_out */
        Stage_parms* sp = *it;
        do_registration_stage (regp, regd, xf_out, xf_in, sp);
    }

    /* JAS 2012.03.29 - for GPUIT Bspline
     * make output match input resolution - not final stage resolution */
    check_output_resolution (xf_out, regd);

    *xf_result = xf_out;
    delete xf_in;
}

void
do_registration (Registration_parms* regp)
{
    Registration_data regd;
    Xform* xf_out = NULL;
    Plm_timer timer1, timer2, timer3;

    /* Start logging */
    logfile_open (regp->log_fn);

    /* Load images */
    // printf ("Performing < %i > registrations.\n", regp->num_jobs);
    for (regp->job_idx=0; regp->job_idx < regp->num_jobs; regp->job_idx++) {

        if (regp->num_jobs > 1) {
            regp->set_job_paths ();
            if (regp->get_fixed_fn() == regp->get_moving_fn()) {
                continue;
            }
        }

        timer1.start();
        regd.load_input_files (regp);
        timer1.stop();
    
        timer2.start();
        do_registration_pure (&xf_out, &regd, regp);
        timer2.stop();

        /* RMK: If no stages, we still generate output (same as input) */
    
        timer3.start();
        save_output (&regd, xf_out, regp->xf_out_fn, regp->xf_out_itk, 
            regp->img_out_fmt, regp->img_out_type, 
            regp->default_value, regp->img_out_fn, 
            regp->vf_out_fn);
        timer3.stop();
    
        delete xf_out;
    
        logfile_open (regp->log_fn);
        logfile_printf (
            "Load:   %g\n"
            "Run:    %g\n"
            "Save:   %g\n"
            "Total:  %g\n",
            (double) timer1.report(),
            (double) timer2.report(),
            (double) timer3.report(),
            (double) timer1.report() + 
            (double) timer2.report() + 
            (double) timer3.report());
    
        /* Done logging */
        logfile_printf ("Finished!\n");
        logfile_close ();
    }
}
