/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "plmbase.h"
#include "plmsys.h"

#include "cxt_extract.h"
#if GDCM_VERSION_1
#include "gdcm1_rtss.h"
#endif
#include "itk_image_load.h"
#include "itk_image_save.h"
#include "plm_image_type.h"
#include "plm_warp.h"
#include "pstring.h"
#include "slice_index.h"
#include "rtds_warp.h"
#include "rtss.h"
#include "simplify_points.h"
#include "ss_img_extract.h"
#include "ss_img_stats.h"
#include "ss_list_io.h"
#include "xio_dose.h"
#include "xio_structures.h"

static void
load_input_files (Rtds *rtds, Plm_file_format file_type, Warp_parms *parms)
{
    if (parms->input_fn.not_empty ()) {
        switch (file_type) {
        case PLM_FILE_FMT_NO_FILE:
            print_and_exit ("Could not open input file %s for read\n",
                (const char*) parms->input_fn);
            break;
        case PLM_FILE_FMT_UNKNOWN:
        case PLM_FILE_FMT_IMG:
            rtds->m_img = plm_image_load_native (parms->input_fn);
            break;
        case PLM_FILE_FMT_DICOM_DIR:
            rtds->load_dicom_dir ((const char*) parms->input_fn);
            break;
        case PLM_FILE_FMT_XIO_DIR:
            rtds->load_xio (
                (const char*) parms->input_fn, &rtds->m_rdd);
            break;
        case PLM_FILE_FMT_DIJ:
            print_and_exit (
                "Warping dij files requires ctatts_in, dif_in files\n");
            break;
#if GDCM_VERSION_1
        case PLM_FILE_FMT_DICOM_RTSS:
            rtds->m_rtss = new Rtss (rtds);
            rtds->m_rtss->load_gdcm_rtss (
                (const char*) parms->input_fn, &rtds->m_rdd);
            break;
        case PLM_FILE_FMT_DICOM_DOSE:
            rtds->m_dose = gdcm1_dose_load (
                0, 
                (const char*) parms->input_fn, 
                (const char*) parms->referenced_dicom_dir);
            break;
#endif
        case PLM_FILE_FMT_CXT:
            rtds->m_rtss = new Rtss (rtds);
            rtds->m_rtss->load_cxt (parms->input_fn, &rtds->m_rdd);
            break;
        case PLM_FILE_FMT_SS_IMG_VEC:
        default:
            print_and_exit (
                "Sorry, don't know how to convert/warp input type %s (%s)\n",
                plm_file_format_string (file_type),
                (const char*) parms->input_fn);
            break;
        }
    }

    if (parms->input_cxt_fn.not_empty()) {
        if (rtds->m_rtss) delete rtds->m_rtss;
        rtds->m_rtss = new Rtss (rtds);
        rtds->m_rtss->load_cxt (parms->input_cxt_fn, &rtds->m_rdd);
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
}

static void
save_ss_img (
    Rtds *rtds, 
    Xform *xf, 
    Plm_image_header *pih, 
    Warp_parms *parms
)
{
    /* labelmap */
    if (parms->output_labelmap_fn.not_empty()) {
        printf ("save_ss_img: save_labelmap\n");
        rtds->m_rtss->save_labelmap (parms->output_labelmap_fn);
    }

    /* ss_img */
    if (parms->output_ss_img_fn.not_empty()) {
        printf ("save_ss_img: save_ss_image\n");
        rtds->m_rtss->save_ss_image (parms->output_ss_img_fn);
    }

    /* list of structure names */
    if (parms->output_ss_list_fn.not_empty()) {
        printf ("save_ss_img: save_ss_list\n");
        rtds->m_rtss->save_ss_list (parms->output_ss_list_fn);
    }

    /* prefix images */
    if (parms->output_prefix.not_empty()) {
        printf ("save_ss_img: save_prefix\n");
        rtds->m_rtss->save_prefix (parms->output_prefix);
    }

    /* prefix fcsv files */
    if (parms->output_prefix_fcsv.not_empty()) {
        printf ("save_ss_img: save_prefix_fcsv\n");
        printf ("save_ss_img: save_prefix_fcsv (%s)\n",
            (const char*) parms->output_prefix_fcsv);
        rtds->m_rtss->save_prefix_fcsv (parms->output_prefix_fcsv);
    }

    /* 3D Slicer color table */
    if (parms->output_colormap_fn.not_empty()) {
        printf ("save_ss_img: save_colormap\n");
        rtds->m_rtss->save_colormap (parms->output_colormap_fn);
    }

    /* cxt */
    if (parms->output_cxt_fn.not_empty()) {
        printf ("save_ss_img: save_cxt\n");
        rtds->m_rtss->save_cxt (&rtds->m_rdd, parms->output_cxt_fn, false);
    }

    /* xio */
    if (parms->output_xio_dirname.not_empty()) {
        logfile_printf ("save_ss_img: save_xio (dirname = %s)\n", 
            (const char*) parms->output_xio_dirname);
        rtds->m_rtss->save_xio (
            rtds->m_xio_transform,
            parms->output_xio_version,
            parms->output_xio_dirname);
    }
}

static void
warp_and_save_ss (
    Rtds *rtds,  
    Xform *xf, 
    Plm_image_header *pih, 
    Warp_parms *parms)
{
    if (!rtds->m_rtss) {
        return;
    }

    /* If we have need to create image outputs, or if we have to 
       warp something, then we need to rasterize the volume */
    /* GCS FIX: If there is an input m_ss_img, we still do this 
       because we might need the labelmap */
    if (parms->output_labelmap_fn.not_empty()
        || parms->output_ss_img_fn.not_empty()
        || parms->xf_in_fn.not_empty()
        || parms->output_prefix.not_empty())
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
        Rtss_polyline_set *cxt = rtds->m_rtss->m_cxt;
        if (parms->xf_in_fn.not_empty()) {
            pih.set_from_gpuit (cxt->rast_dim, cxt->rast_offset, 
                cxt->rast_spacing, 0);
        } else {
            pih.set_from_gpuit (cxt->m_dim, cxt->m_offset, cxt->m_spacing, 0);
        }
        printf ("Warp_and_save_ss: m_rtss->rasterize\n");
        rtds->m_rtss->rasterize (&pih,
            parms->output_labelmap_fn.not_empty(),
            parms->xor_contours);
    }

    /* Do the warp */
    if (parms->xf_in_fn.not_empty()) {
        printf ("Warp_and_save_ss: m_rtss->warp\n");
        rtds->m_rtss->warp (xf, pih, parms);
    }

    /* If we are warping, re-extract polylines into cxt */
    /* GCS FIX: This is only necessary if we are outputting polylines. 
       Otherwise it is wasting users time. */
    if (parms->xf_in_fn.not_empty()) {
        printf ("Warp_and_save_ss: m_rtss->cxt_re_extract\n");
        rtds->m_rtss->cxt_re_extract ();
    }

    /* If we need to reduce the number of points (aka if simplify-perc 
       was set), purge the excessive points...*/
    if (parms->simplify_perc > 0. && parms->simplify_perc < 100.) {
        printf ("Warp_and_save_ss: do_simplify\n");
        do_simplify(rtds, parms->simplify_perc);
    }

    /* Save non-dicom formats, such as mha, cxt, xio */
    printf ("Warp_and_save_ss: save_ss_img\n");
    save_ss_img (rtds, xf, pih, parms);
}

void
rtds_warp (Rtds *rtds, Plm_file_format file_type, Warp_parms *parms)
{
    DeformationFieldType::Pointer vf = DeformationFieldType::New();
    Xform xform;
    Plm_image_header pih;

    /* Load referenced DICOM directory */

    if (parms->referenced_dicom_dir.not_empty()) {
        logfile_printf ("Loading RDD\n");
        rtds->load_rdd ((const char*) parms->referenced_dicom_dir);
    } else {
        /* GCS: 2011-09-05.  I think it is better to ask the user
           to explicitly choose a referenced dicom dir than load
           a directory by default. */
#if defined (commentout)
        /* Look for referenced CT in input directory */
        if (parms->input_fn.not_empty()) {
            logfile_printf ("Loading RDD\n");
            char* dirname = file_util_dirname ((const char*) parms->input_fn);
            rtds->load_rdd (dirname);
            free (dirname);
        }
#endif
    }

    /* Load input file(s) */
    load_input_files (rtds, file_type, parms);

    /* Set user-supplied metadata (overrides metadata in input files) */
    rtds->set_user_metadata (parms->m_metadata);

    /* Load transform */
    if (parms->xf_in_fn.not_empty()) {
        logfile_printf ("Loading xform (%s)\n", (const char*) parms->xf_in_fn);
        xform_load (&xform, (const char*) parms->xf_in_fn);
    }

    /* Try to guess the proper dimensions and spacing for output image */
    if (parms->fixed_img_fn.not_empty()) {
        /* use the spacing of user-supplied fixed image */
        printf ("Setting PIH from FIXED\n");
        FloatImageType::Pointer fixed = itk_image_load_float (
            parms->fixed_img_fn, 0);
        pih.set_from_itk_image (fixed);
    } else if (xform.m_type == XFORM_ITK_VECTOR_FIELD) {
        /* use the spacing from input vector field */
        pih.set_from_itk_image (xform.get_itk_vf());
    } else if (xform.m_type == XFORM_GPUIT_BSPLINE) {
        /* use the spacing from input bxf file */
        printf ("Setting PIH from XFORM\n");
        pih.set_from_gpuit_bspline (xform.get_gpuit_bsp());
    } else if (rtds->m_rdd.m_loaded) {
        /* use spacing from referenced CT */
        printf ("Setting PIH from RDD\n");
        Plm_image_header::clone (&pih, &rtds->m_rdd.m_pih);
    } else if (rtds->m_img) {
        /* use the spacing of the input image */
        printf ("Setting PIH from M_IMG\n");
        pih.set_from_plm_image (rtds->m_img);
    } else if (rtds->m_rtss && rtds->m_rtss->m_ss_img) {
        /* use the spacing of the input image */
        printf ("Setting PIH from M_SS_IMG\n");
        pih.set_from_plm_image (rtds->m_rtss->m_ss_img);
    }
    else if (rtds->m_rtss &&
        rtds->m_rtss->m_cxt && 
        rtds->m_rtss->m_cxt->have_geometry) {
        /* use the spacing of the structure set */
        pih.set_from_gpuit (
            rtds->m_rtss->m_cxt->m_dim, 
            rtds->m_rtss->m_cxt->m_offset, 
            rtds->m_rtss->m_cxt->m_spacing, 
            0);
    } else if (rtds->m_dose) {
        /* use the spacing of dose */
        printf ("Setting PIH from DOSE\n");
        pih.set_from_plm_image (rtds->m_dose);
    } else if (rtds->m_rtss && rtds->m_rtss->m_cxt) {
        /* we have structure set, but without geometry.  use 
           heuristics to find a good geometry for rasterization */
        rtds->m_rtss->find_rasterization_geometry (&pih);
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

    printf ("PIH is:\n");
    pih.print ();

    /* Warp the image and create vf */
    if (rtds->m_img 
        && parms->xf_in_fn.not_empty()
        && (parms->output_img_fn.not_empty()
            || parms->output_vf_fn.not_empty()
            || parms->output_dicom.not_empty()))
    {
        Plm_image *im_out;
        im_out = new Plm_image;
        printf ("Rtds_warp: Warping m_img\n");
        plm_warp (im_out, &vf, &xform, &pih, rtds->m_img, parms->default_val, 
            parms->use_itk, parms->interp_lin);
        delete rtds->m_img;
        rtds->m_img = im_out;
    }

    /* Save output image */
    if (parms->output_img_fn.not_empty() && rtds->m_img) {
        printf ("Rtds_warp: Saving m_img (%s)\n",
            (const char*) parms->output_img_fn);
        rtds->m_img->convert_and_save (
            (const char*) parms->output_img_fn, 
            parms->output_type);
    }

    /* Warp the dose image */
    if (rtds->m_dose
        && parms->xf_in_fn.not_empty()
        && (parms->output_dose_img_fn.not_empty()
            || parms->output_xio_dirname.not_empty()
            || parms->output_dicom.not_empty()))
    {
        printf ("Rtds_warp: Warping dose\n");
        Plm_image *im_out;
        im_out = new Plm_image;
        plm_warp (im_out, 0, &xform, &pih, rtds->m_dose, 0, 
            parms->use_itk, 1);
        delete rtds->m_dose;
        rtds->m_dose = im_out;
    }

    /* Scale the dose image */
    if (rtds->m_dose && parms->have_dose_scale) {
        volume_scale (rtds->m_dose->gpuit_float(), parms->dose_scale);
    }

    /* Save output dose image */
    if (parms->output_dose_img_fn.not_empty() && rtds->m_dose)
    {
        printf ("Rtds_warp: Saving dose image (%s)\n", 
            (const char*) parms->output_dose_img_fn);
        rtds->m_dose->convert_and_save (
            (const char*) parms->output_dose_img_fn, 
            parms->output_type);
    }

    /* Save output XiO dose */
    if (parms->output_xio_dirname.not_empty()
        && rtds->m_xio_dose_input
        && rtds->m_dose)
    {
        Pstring fn;

        printf ("Rtds_warp: Saving xio dose.\n");
        fn.format ("%s/%s", (const char*) parms->output_xio_dirname, "dose");
        xio_dose_save (
            rtds->m_dose,
            &(rtds->m_meta),
            rtds->m_xio_transform, 
            (const char*) fn, 
            rtds->m_xio_dose_input);
    }

    /* Save output vector field */
    if (parms->xf_in_fn.not_empty() 
        && parms->output_vf_fn.not_empty())
    {
        printf ("Rtds_warp: Saving vf.\n");
        itk_image_save (vf, (const char*) parms->output_vf_fn);
    }

    /* Preprocess structure sets */
    if (rtds->m_rtss) {

        /* Convert ss_img to cxt */
        printf ("Rtds_warp: Convert ss_img to cxt.\n");
        rtds->m_rtss->convert_ss_img_to_cxt ();

        /* Delete empty structures */
        if (parms->prune_empty) {
            printf ("Rtds_warp: Prune empty structures.\n");
            rtds->m_rtss->prune_empty ();
        }

        /* Set the DICOM reference info -- this sets the internal geometry 
           of the ss_image so we rasterize on the same slices as the CT? */
        printf ("Rtds_warp: Apply dicom_dir.\n");
        rtds->m_rtss->apply_dicom_dir (&rtds->m_rdd);
        
        /* Set the output geometry */
        printf ("Rtds_warp: Set geometry from PIH.\n");
        rtds->m_rtss->set_geometry_from_plm_image_header (&pih);

        /* Set rasterization geometry */
        printf ("Rtds_warp: Set rasterization geometry.\n");
        rtds->m_rtss->m_cxt->set_rasterization_geometry ();
    }

    /* Warp and save structure set (except dicom) */
    printf ("Rtds_warp: warp and save ss.\n");
    warp_and_save_ss (rtds, &xform, &pih, parms);

    /* Save dicom */
    if (parms->output_dicom.not_empty()) {
        printf ("Rtds_warp: Save dicom.\n");
        rtds->save_dicom ((const char*) parms->output_dicom);
    }
}