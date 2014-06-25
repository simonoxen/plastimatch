/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"

#if GDCM_VERSION_1
#include "gdcm1_dose.h"
#endif
#include "file_util.h"
#include "itk_image_load.h"
#include "itk_image_save.h"
#include "itk_image_type.h"
#include "logfile.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_warp.h"
#include "print_and_exit.h"
#include "pstring.h"
#include "rt_study.h"
#include "rt_study_metadata.h"
#include "rtds_warp.h"
#include "rtss.h"
#include "segmentation.h"
#include "simplify_points.h"
#include "warp_parms.h"
#include "volume.h"
#include "xform.h"
#include "xio_dose.h"

static void
load_input_files (Rt_study *rtds, Plm_file_format file_type, Warp_parms *parms)
{
    if (parms->input_fn.not_empty ()) {
        rtds->load (parms->input_fn.c_str(), file_type);
    }

    if (parms->input_cxt_fn.not_empty()) {
        rtds->load_cxt (parms->input_cxt_fn);
    }

    if (parms->input_prefix.not_empty()) {
        rtds->load_prefix (parms->input_prefix);
    }

    if (parms->input_ss_img_fn.not_empty()) {
        if (!file_exists (parms->input_ss_img_fn)) {
            print_and_exit ("Error: cannot open file %s for read\n",
                (const char*) parms->input_ss_img_fn);
        }
        rtds->load_ss_img (
            (const char*) parms->input_ss_img_fn, 
            (const char*) parms->input_ss_list_fn);
    }

    if (parms->input_dose_img_fn.not_empty()) {
        rtds->load_dose_img ((const char*) parms->input_dose_img_fn);
    }

    if (parms->input_dose_xio_fn.not_empty()) {
        rtds->load_dose_xio ((const char*) parms->input_dose_xio_fn);
    }

    if (parms->input_dose_ast_fn.not_empty()) {
        rtds->load_dose_astroid ((const char*) parms->input_dose_ast_fn);
    }

    if (parms->input_dose_mc_fn.not_empty()) {
        rtds->load_dose_mc ((const char*) parms->input_dose_mc_fn);
    }

    if (!rtds->have_image() && !rtds->have_rtss() && !rtds->have_dose()) {
        print_and_exit ("Sorry, could not load input as any known type.\n");
    }
}

static void
save_ss_img (
    Rt_study *rtds, 
    const Xform *xf, 
    Plm_image_header *pih, 
    Warp_parms *parms
)
{
    Segmentation::Pointer seg = rtds->get_rtss();

    /* labelmap */
    if (parms->output_labelmap_fn.not_empty()) {
        lprintf ("save_ss_img: save_labelmap\n");
        seg->save_labelmap (parms->output_labelmap_fn);
    }

    /* ss_img */
    if (parms->output_ss_img_fn.not_empty()) {
        lprintf ("save_ss_img: save_ss_image\n");
        seg->save_ss_image (parms->output_ss_img_fn);
    }

    /* list of structure names */
    if (parms->output_ss_list_fn.not_empty()) {
        lprintf ("save_ss_img: save_ss_list\n");
        seg->save_ss_list (parms->output_ss_list_fn);
    }

    /* prefix images */
    if (parms->output_prefix != "") {
        lprintf ("save_ss_img: save_prefix\n");
        seg->save_prefix (parms->output_prefix, parms->prefix_format);
    }

    /* prefix fcsv files */
    if (parms->output_prefix_fcsv.not_empty()) {
        lprintf ("save_ss_img: save_prefix_fcsv\n");
        lprintf ("save_ss_img: save_prefix_fcsv (%s)\n",
            (const char*) parms->output_prefix_fcsv);
        seg->save_prefix_fcsv (parms->output_prefix_fcsv);
    }

    /* 3D Slicer color table */
    if (parms->output_colormap_fn.not_empty()) {
        lprintf ("save_ss_img: save_colormap\n");
        seg->save_colormap (parms->output_colormap_fn);
    }

    /* cxt */
    if (parms->output_cxt_fn.not_empty()) {
        lprintf ("save_ss_img: save_cxt\n");
        seg->save_cxt (rtds->get_rt_study_metadata (),
            parms->output_cxt_fn, false);
    }

    /* xio */
    if (parms->output_xio_dirname.not_empty()) {
        lprintf ("save_ss_img: save_xio (dirname = %s)\n", 
            (const char*) parms->output_xio_dirname);
        seg->save_xio (
            rtds->get_xio_ct_transform(),
            parms->output_xio_version,
            parms->output_xio_dirname);
    }
}

static void
warp_and_save_ss (
    Rt_study *rtds,  
    const Xform::Pointer& xf, 
    Plm_image_header *pih, 
    Warp_parms *parms)
{
    if (!rtds->have_rtss()) {
        return;
    }

    Segmentation::Pointer seg = rtds->get_rtss();

    /* If we have need to create image outputs, or if we have to 
       warp something, then we need to rasterize the volume */
    /* GCS FIX: If there is an input m_ss_img, we still do this 
       because we might need the labelmap */
    if (parms->output_labelmap_fn.not_empty()
        || parms->output_ss_img_fn.not_empty()
        || parms->xf_in_fn.not_empty()
        || parms->output_prefix != "")
    {

        /* In the following cases, we should use the default 
           rasterization geometry (i.e. cxt->rast_xxx):
           (a) Warping
           (b) No known output geometry
           (c) Output geometry doesn't match slice locations

           GCS FIX: Only case (a) is handled.

           In the other cases we can directly rasterize to the output 
           geometry.
        */
        Plm_image_header pih;
        Rtss *cxt = seg->get_structure_set_raw ();
        if (parms->xf_in_fn.not_empty()) {
            pih.set_from_gpuit (cxt->rast_dim, cxt->rast_offset, 
                cxt->rast_spacing, 0);
        } else {
            pih.set_from_gpuit (cxt->m_dim, cxt->m_offset, cxt->m_spacing, 0);
        }
        lprintf ("Warp_and_save_ss: seg->rasterize\n");
        seg->rasterize (&pih,
            parms->output_labelmap_fn.not_empty(),
            parms->xor_contours);
    }

    /* Do the warp */
    if (parms->xf_in_fn.not_empty()) {
        lprintf ("Warp_and_save_ss: seg->warp\n");
        seg->warp (xf, pih, parms);
    }

    /* If we are warping, re-extract polylines into cxt */
    /* GCS FIX: This is only necessary if we are outputting polylines. 
       Otherwise it is wasting users time. */
    if (parms->xf_in_fn.not_empty()) {
        lprintf ("Warp_and_save_ss: seg->cxt_re_extract\n");
        seg->cxt_re_extract ();
    }

    /* If we need to reduce the number of points (aka if simplify-perc 
       was set), purge the excessive points...*/
    if (parms->simplify_perc > 0. && parms->simplify_perc < 100.) {
        lprintf ("Warp_and_save_ss: do_simplify\n");
        do_simplify(rtds, parms->simplify_perc);
    }

    /* Save non-dicom formats, such as mha, cxt, xio */
    lprintf ("Warp_and_save_ss: save_ss_img\n");
    save_ss_img (rtds, xf.get(), pih, parms);
}

void
rtds_warp (Rt_study *rtds, Plm_file_format file_type, Warp_parms *parms)
{
    DeformationFieldType::Pointer vf = DeformationFieldType::New();
    Xform::Pointer xform = Xform::New ();
    Plm_image_header pih;

    /* Load referenced DICOM directory */
    if (parms->referenced_dicom_dir.not_empty()) {
        lprintf ("Loading RDD\n");
        rtds->load_rdd ((const char*) parms->referenced_dicom_dir);
        lprintf ("Loading RDD complete\n");
    }

    /* Set user-supplied metadata also prior to loading files,
       because the user supplied patient position is needed to load XiO data */
    rtds->set_user_metadata (parms->m_metadata);

    /* Load input file(s) */
    load_input_files (rtds, file_type, parms);

    /* Set user-supplied metadata (overrides metadata in input files) */
    rtds->set_user_metadata (parms->m_metadata);

    /* Load transform */
    if (parms->xf_in_fn.not_empty()) {
        lprintf ("Loading xform (%s)\n", (const char*) parms->xf_in_fn);
        xform = xform_load ((const char*) parms->xf_in_fn);
    }

    /* Try to guess the proper dimensions and spacing for output image */
    Xform_type xform_type = xform->get_type ();
    if (parms->fixed_img_fn.not_empty()) {
        /* use the spacing of user-supplied fixed image */
        lprintf ("Setting PIH from FIXED\n");
        FloatImageType::Pointer fixed = itk_image_load_float (
            parms->fixed_img_fn, 0);
        pih.set_from_itk_image (fixed);
    } else if (xform_type == XFORM_ITK_VECTOR_FIELD) {
        /* use the spacing from input vector field */
        lprintf ("Setting PIH from VF\n");
        pih.set_from_itk_image (xform->get_itk_vf());
    } else if (xform_type == XFORM_GPUIT_BSPLINE) {
        /* use the spacing from input bxf file */
        lprintf ("Setting PIH from XFORM\n");
        pih.set_from_gpuit_bspline (xform->get_gpuit_bsp());
    } else if (rtds->get_rt_study_metadata()->slice_list_complete()) {
        /* use spacing from referenced CT */
        lprintf ("Setting PIH from RDD\n");
        Plm_image_header::clone (&pih, 
            rtds->get_rt_study_metadata()->get_image_header());
    } else if (rtds->have_image()) {
        /* use the spacing of the input image */
        lprintf ("Setting PIH from M_IMG\n");
        pih.set_from_plm_image (rtds->get_image().get());
    } else if (rtds->have_rtss() && rtds->get_rtss()->have_ss_img()) {
        /* use the spacing of the input image */
        lprintf ("Setting PIH from M_SS_IMG\n");
        pih.set_from_plm_image (rtds->get_rtss()->get_ss_img());
    }
    else if (rtds->have_rtss() &&
        /* use the spacing of the structure set */
        rtds->get_rtss()->have_structure_set() && 
        rtds->get_rtss()->get_structure_set()->have_geometry) {
        pih.set_from_gpuit (
            rtds->get_rtss()->get_structure_set()->m_dim, 
            rtds->get_rtss()->get_structure_set()->m_offset, 
            rtds->get_rtss()->get_structure_set()->m_spacing, 
            0);
    } else if (rtds->has_dose()) {
        /* use the spacing of dose */
        lprintf ("Setting PIH from DOSE\n");
        pih.set_from_plm_image (rtds->get_dose());
    } else if (rtds->have_rtss() && rtds->get_rtss()->have_structure_set()) {
        /* we have structure set, but without geometry.  use 
           heuristics to find a good geometry for rasterization */
        rtds->get_rtss()->find_rasterization_geometry (&pih);
    } else {
        /* use some generic default parameters */
        plm_long dim[3] = { 500, 500, 500 };
        float origin[3] = { -249.5, -249.5, -249.5 };
        float spacing[3] = { 1., 1., 1. };
        pih.set_from_gpuit (dim, origin, spacing, 0);
    }

    if (parms->m_have_dim) {
        pih.set_dim (parms->m_dim);
    }
    if (parms->m_have_origin) {
        pih.set_origin (parms->m_origin);
    }
    if (parms->m_have_spacing) {
        pih.set_spacing (parms->m_spacing);
    }
    if (parms->m_have_direction_cosines) {
        /* GCS FIX: This will do illogical things unless the user 
           sets the origin properly */
        pih.set_direction_cosines (parms->m_dc);
    }

    lprintf ("PIH is:\n");
    pih.print ();

    /* Warp the image and create vf */
    if (rtds->have_image()
        && parms->xf_in_fn.not_empty()
        && (parms->output_img_fn.not_empty()
            || parms->output_vf_fn.not_empty()
            || parms->output_dicom.not_empty()))
    {
        Plm_image::Pointer im_out = Plm_image::New();
        lprintf ("Rt_study_warp: Warping m_img\n");
        plm_warp (im_out, &vf, xform, &pih, rtds->get_image(), 
            parms->default_val, parms->use_itk, parms->interp_lin);
        rtds->set_image (im_out);
    }

    /* Save output image */
    if (parms->output_img_fn.not_empty() && rtds->have_image()) {
        lprintf ("Rt_study_warp: Saving m_img (%s)\n",
            (const char*) parms->output_img_fn);
        rtds->get_image()->convert_and_save (
            (const char*) parms->output_img_fn, 
            parms->output_type);
    }

    /* Warp the dose image */
    if (rtds->has_dose()
        && parms->xf_in_fn.not_empty()
        && (parms->output_dose_img_fn.not_empty()
            || parms->output_xio_dirname.not_empty()
            || parms->output_dicom.not_empty()))
    {
        lprintf ("Rt_study_warp: Warping dose\n");
        Plm_image::Pointer im_out = Plm_image::New();
        plm_warp (im_out, 0, xform, &pih, rtds->get_dose(), 0, 
            parms->use_itk, 1);
        rtds->set_dose (im_out);
    }

    /* Scale the dose image */
    if (rtds->has_dose() && parms->have_dose_scale) {
        rtds->get_dose_volume_float()->scale_inplace (parms->dose_scale);
    }

    /* Save output dose image */
    if (parms->output_dose_img_fn.not_empty() && rtds->has_dose())
    {
        lprintf ("Rt_study_warp: Saving dose image (%s)\n", 
            (const char*) parms->output_dose_img_fn);
#if defined (commentout)
        rtds->m_dose->convert_and_save (
            (const char*) parms->output_dose_img_fn, 
            parms->output_type);
#endif
        rtds->save_dose (
            (const char*) parms->output_dose_img_fn, 
            parms->output_type);
    }

    /* Save output XiO dose */
    if (parms->output_xio_dirname.not_empty()
        && rtds->get_xio_dose_filename() != ""
        && rtds->has_dose())
    {
        Pstring fn;

        lprintf ("Rt_study_warp: Saving xio dose.\n");
        fn.format ("%s/%s", (const char*) parms->output_xio_dirname, "dose");
        xio_dose_save (
            rtds->get_dose(),
            rtds->get_metadata(), 
            rtds->get_xio_ct_transform(),
            (const char*) fn, 
            rtds->get_xio_dose_filename().c_str());
    }

    /* Save output vector field */
    if (parms->xf_in_fn.not_empty() 
        && parms->output_vf_fn.not_empty())
    {
        lprintf ("Rt_study_warp: Saving vf.\n");
        itk_image_save (vf, (const char*) parms->output_vf_fn);
    }

    /* Preprocess structure sets */
    if (rtds->have_rtss()) {
        Segmentation::Pointer seg = rtds->get_rtss();

        /* Convert ss_img to cxt */
        lprintf ("Rt_study_warp: Convert ss_img to cxt.\n");
        seg->convert_ss_img_to_cxt ();

        /* Delete empty structures */
        if (parms->prune_empty) {
            lprintf ("Rt_study_warp: Prune empty structures.\n");
            seg->prune_empty ();
        }

        /* Set the DICOM reference info -- this sets the internal geometry 
           of the ss_image so we rasterize on the same slices as the CT? */
        lprintf ("Rt_study_warp: Apply dicom_dir.\n");
        seg->apply_dicom_dir (rtds->get_rt_study_metadata());
        
        /* Set the output geometry */
        lprintf ("Rt_study_warp: Set geometry from PIH.\n");
        seg->set_geometry (&pih);

        /* Set rasterization geometry */
        lprintf ("Rt_study_warp: Set rasterization geometry.\n");
        seg->get_structure_set()->set_rasterization_geometry ();
    }

    /* Warp and save structure set (except dicom) */
    lprintf ("Rt_study_warp: warp and save ss.\n");
    warp_and_save_ss (rtds, xform, &pih, parms);

    /* Save dicom */
    if (parms->output_dicom.not_empty()) {
        lprintf ("Rt_study_warp: Save dicom.\n");
        rtds->save_dicom ((const char*) parms->output_dicom);
    }
}
