/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "bstring_util.h"
#include "cxt_extract.h"
#include "cxt_to_mha.h"
#include "file_util.h"
#include "gdcm_dose.h"
#include "gdcm_rtss.h"
#include "itk_image_save.h"
#include "logfile.h"
#include "plm_image_type.h"
#include "plm_warp.h"
#include "referenced_dicom_dir.h"
#include "rtds_dicom.h"
#include "rtds_warp.h"
#include "simplify_points.h"
#include "ss_img_extract.h"
#include "ss_list_io.h"
#include "xio_dose.h"
#include "xio_structures.h"

static void
load_input_files (Rtds *rtds, Plm_file_format file_type, Warp_parms *parms)
{
    if (bstring_not_empty (parms->input_fn)) {
	switch (file_type) {
	case PLM_FILE_FMT_NO_FILE:
	    print_and_exit ("Could not open input file %s for read\n",
		(const char*) parms->input_fn);
	    break;
	case PLM_FILE_FMT_UNKNOWN:
	case PLM_FILE_FMT_IMG:
	    rtds->m_img = plm_image_load_native (parms->input_fn);
	    if (parms->patient_pos == PATIENT_POSITION_UNKNOWN 
		&& bstring_not_empty (parms->referenced_dicom_dir))
	    {
		rtds_patient_pos_from_dicom_dir (rtds, parms->referenced_dicom_dir);
	    } else {
		rtds->m_img->m_patient_pos = parms->patient_pos;
	    }
	    break;
	case PLM_FILE_FMT_DICOM_DIR:
	    rtds->load_dicom_dir ((const char*) parms->input_fn);
	    break;
	case PLM_FILE_FMT_XIO_DIR:
	    rtds->load_xio (
		(const char*) parms->input_fn, 
		(const char*) parms->referenced_dicom_dir, 
		parms->patient_pos);
	    break;
	case PLM_FILE_FMT_DIJ:
	    print_and_exit (
		"Warping dij files requires ctatts_in, dif_in files\n");
	    break;
	case PLM_FILE_FMT_DICOM_RTSS:
	    rtds->m_ss_image = new Ss_image;
	    rtds->m_ss_image->load_gdcm_rtss (
		(const char*) parms->input_fn, &rtds->m_rdd);
	    break;
	case PLM_FILE_FMT_DICOM_DOSE:
	    rtds->m_dose = gdcm_dose_load (
		0, 
		(const char*) parms->input_fn, 
		(const char*) parms->referenced_dicom_dir);
	    break;
	case PLM_FILE_FMT_CXT:
	    rtds->m_ss_image = new Ss_image;
	    rtds->m_ss_image->load_cxt (parms->input_fn, &rtds->m_rdd);
	    break;
	case PLM_FILE_FMT_SS_IMG_4D:
	default:
	    print_and_exit (
		"Sorry, don't know how to convert/warp input type %s (%s)\n",
		plm_file_format_string (file_type),
		(const char*) parms->input_fn);
	    break;
	}
    }

    if (bstring_not_empty (parms->referenced_dicom_dir)) {
	logfile_printf ("Loading RDD\n");
	rtds->load_rdd ((const char*) parms->referenced_dicom_dir);
    } else {
	/* Look for referenced CT in input directory */
	if (bstring_not_empty (parms->input_fn)) {
	    logfile_printf ("Loading RDD\n");
	    char* dirname = file_util_dirname ((const char*) parms->input_fn);
	    rtds->load_rdd (dirname);
	    free (dirname);
	}
    }

    if (bstring_not_empty (parms->input_cxt_fn)) {
	if (rtds->m_ss_image) delete rtds->m_ss_image;
	rtds->m_ss_image = new Ss_image;
	rtds->m_ss_image->load_cxt (parms->input_cxt_fn, &rtds->m_rdd);
    }

    if (bstring_not_empty (parms->input_ss_img_fn)) {
	rtds->load_ss_img (
	    (const char*) parms->input_ss_img_fn, 
	    (const char*) parms->input_ss_list_fn);
    }

    if (bstring_not_empty (parms->input_dose_img_fn)) {
	rtds->load_dose_img ((const char*) parms->input_dose_img_fn);
    }

    if (bstring_not_empty (parms->input_dose_xio_fn)) {
	rtds->load_dose_xio ((const char*) parms->input_dose_xio_fn,
	    parms->patient_pos);
    }

    if (bstring_not_empty (parms->input_dose_ast_fn)) {
	rtds->load_dose_astroid ((const char*) parms->input_dose_ast_fn,
	    parms->patient_pos);
    }

    if (bstring_not_empty (parms->input_dose_mc_fn)) {
	rtds->load_dose_mc ((const char*) parms->input_dose_mc_fn,
	    parms->patient_pos);
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
    if (bstring_not_empty (parms->output_labelmap_fn)) {
	rtds->m_ss_image->save_labelmap (parms->output_labelmap_fn);
    }

    /* ss_img */
    if (bstring_not_empty (parms->output_ss_img_fn)) {
	rtds->m_ss_image->save_ss_image (parms->output_ss_img_fn);
    }

    /* list of structure names */
    if (bstring_not_empty (parms->output_ss_list_fn)) {
	rtds->m_ss_image->save_ss_list (parms->output_ss_list_fn);
    }

    /* prefix images */
    if (bstring_not_empty (parms->output_prefix)) {
	rtds->m_ss_image->save_prefix (parms->output_prefix);
    }

    /* 3D Slicer color table */
    if (bstring_not_empty (parms->output_colormap_fn)) {
	rtds->m_ss_image->save_colormap (parms->output_colormap_fn);
    }

    /* cxt */
    if (bstring_not_empty (parms->output_cxt_fn)) {
	rtds->m_ss_image->save_cxt (&rtds->m_rdd, parms->output_cxt_fn, false);
    }

    /* xio */
    if (bstring_not_empty (parms->output_xio_dirname)) {
	logfile_printf ("Output xio dirname = %s\n", 
	    (const char*) parms->output_xio_dirname);
	rtds->m_ss_image->save_xio (
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
    if (!rtds->m_ss_image) {
	return;
    }

    /* If we have need to create image outputs, or if we have to 
       warp something, then we need to rasterize the volume */
    /* GCS FIX: If there is an input m_ss_img, we still do this 
       because we might need the labelmap */
    if (bstring_not_empty (parms->output_labelmap_fn)
	|| bstring_not_empty (parms->output_ss_img_fn)
	|| bstring_not_empty (parms->xf_in_fn)
	|| bstring_not_empty (parms->output_prefix))
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
	Rtss_polyline_set *cxt = rtds->m_ss_image->m_cxt;
	if (bstring_not_empty (parms->xf_in_fn)) {
	    pih.set_from_gpuit (cxt->rast_offset, 
		cxt->rast_spacing, cxt->rast_dim, 0);
	} else {
	    pih.set_from_gpuit (cxt->m_offset, cxt->m_spacing, cxt->m_dim, 0);
	}

	rtds->m_ss_image->rasterize (&pih);
    }

    /* Do the warp */
    /* GCS FIX: This is inefficient.  We don't need to warp labelmap if 
       not included in output. */
    if (bstring_not_empty (parms->xf_in_fn)) {
	rtds->m_ss_image->warp (xf, pih, parms);
    }

    /* If we are warping, re-extract polylines into cxt */
    /* GCS FIX: This is only necessary if we are outputting polylines. 
       Otherwise it is wasting users time. */
    if (bstring_not_empty (parms->xf_in_fn)) {
	rtds->m_ss_image->cxt_re_extract ();
    }

    /* If we need to reduce the number of points (aka if simplify-perc was set), */
    /* purge the excessive points...*/
    if (parms->simplify_perc >0 && parms->simplify_perc<100) {
	do_simplify(rtds,parms->simplify_perc);
    }

    /* Save non-dicom formats, such as mha, cxt, xio */
    save_ss_img (rtds, xf, pih, parms);
}

void
rtds_warp (Rtds *rtds, Plm_file_format file_type, Warp_parms *parms)
{
    DeformationFieldType::Pointer vf = DeformationFieldType::New();
    Xform xform;
    Plm_image_header pih;

    /* Load input file(s) */
    load_input_files (rtds, file_type, parms);

    /* Load transform */
    if (bstring_not_empty (parms->xf_in_fn)) {
	logfile_printf ("Loading xform (%s)\n", (const char*) parms->xf_in_fn);
	xform_load (&xform, (const char*) parms->xf_in_fn);
    }

    /* Try to guess the proper dimensions and spacing for output image */
    if (bstring_not_empty (parms->fixed_im_fn)) {
	/* use the spacing of user-supplied fixed image */
	printf ("Setting PIH from FIXED\n");
	FloatImageType::Pointer fixed = itk_image_load_float (
	    parms->fixed_im_fn, 0);
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
    } else if (rtds->m_ss_image && rtds->m_ss_image->m_ss_img) {
	/* use the spacing of the input image */
	printf ("Setting PIH from M_SS_IMG\n");
	pih.set_from_plm_image (rtds->m_ss_image->m_ss_img);
    }
    else if (rtds->m_ss_image &&
	rtds->m_ss_image->m_cxt && 
	rtds->m_ss_image->m_cxt->have_geometry) {
	/* use the spacing of the structure set */
	pih.set_from_gpuit (
	    rtds->m_ss_image->m_cxt->m_offset, 
	    rtds->m_ss_image->m_cxt->m_spacing, 
	    rtds->m_ss_image->m_cxt->m_dim, 0);
    } else if (rtds->m_dose) {
	/* use the spacing of dose */
	printf ("Setting PIH from DOSE\n");
	pih.set_from_plm_image (rtds->m_dose);
    } else if (rtds->m_ss_image && rtds->m_ss_image->m_cxt) {
	/* we have structure set, but without geometry.  use 
	   heuristics to find a good geometry for rasterization */
	rtds->m_ss_image->find_rasterization_geometry (&pih);
    } else {
	/* use some generic default parameters */
	int dims[3] = { 500, 500, 500 };
	float offset[3] = { -249.5, -249.5, -249.5 };
	float spacing[3] = { 1., 1., 1. };
	pih.set_from_gpuit (offset, spacing, dims, 0);
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

    printf ("PIH is:\n");
    pih.print ();

    /* Warp the image and create vf */
    if (rtds->m_img 
	&& bstring_not_empty (parms->xf_in_fn)
	&& (bstring_not_empty (parms->output_img_fn)
	    || bstring_not_empty (parms->output_vf_fn)))
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
    if (bstring_not_empty (parms->output_img_fn) && rtds->m_img) {
	printf ("Rtds_warp: Saving m_img (%s)\n",
	    (const char*) parms->output_img_fn);
	rtds->m_img->convert_and_save (
	    (const char*) parms->output_img_fn, 
	    parms->output_type);
    }

    /* Warp the dose image */
    if (rtds->m_dose
	&& bstring_not_empty (parms->xf_in_fn)
	&& (bstring_not_empty (parms->output_dose_img_fn)
	    || bstring_not_empty (parms->output_xio_dirname)
	    || bstring_not_empty (parms->output_dicom)))
    {
	printf ("Rtds_warp: Warping dose\n");
	Plm_image *im_out;
	im_out = new Plm_image;
	plm_warp (im_out, 0, &xform, &pih, rtds->m_dose, 0, 
	    parms->use_itk, 1);
	delete rtds->m_dose;
	rtds->m_dose = im_out;
    }

    /* Save output dose image */
    if (bstring_not_empty (parms->output_dose_img_fn) && rtds->m_dose)
    {
	printf ("Rtds_warp: Saving dose image (%s)\n", 
	    (const char*) parms->output_dose_img_fn);
	rtds->m_dose->convert_and_save (
	    (const char*) parms->output_dose_img_fn, 
	    parms->output_type);
    }

    /* Save output XiO dose */
    if (bstring_not_empty (parms->output_xio_dirname)
	&& rtds->m_xio_dose_input
	&& rtds->m_dose)
    {
	CBString fn;

	printf ("Rtds_warp: Saving xio dose.\n");
	fn.format ("%s/%s", (const char*) parms->output_xio_dirname, "dose");
	xio_dose_save (
	    rtds->m_dose, 
	    rtds->m_xio_transform, 
	    (const char*) fn, 
	    rtds->m_xio_dose_input);
    }

    /* Save output vector field */
    if (bstring_not_empty (parms->xf_in_fn) 
	&& bstring_not_empty (parms->output_vf_fn))
    {
	printf ("Rtds_warp: Saving vf.\n");
	itk_image_save (vf, (const char*) parms->output_vf_fn);
    }

    /* Preprocess structure sets */
    if (rtds->m_ss_image) {

	/* Convert ss_img to cxt */
	printf ("Rtds_warp: Convert ss_img to cxt.\n");
	rtds->m_ss_image->convert_ss_img_to_cxt ();

	/* Delete empty structures */
	if (parms->prune_empty) {
	    printf ("Rtds_warp: Prune empty structures.\n");
	    rtds->m_ss_image->prune_empty ();
	}

	/* Set the DICOM reference info -- this sets the internal geometry 
	   of the ss_image so we rasterize on the same slices as the CT? */
	printf ("Rtds_warp: Apply dicom_dir.\n");
	rtds->m_ss_image->apply_dicom_dir (&rtds->m_rdd);
	
	/* Set the output geometry */
	printf ("Rtds_warp: Set geometry from PIH.\n");
	rtds->m_ss_image->set_geometry_from_plm_image_header (&pih);

	/* Set rasterization geometry */
	printf ("Rtds_warp: Set rasterization geometry.\n");
	rtds->m_ss_image->m_cxt->set_rasterization_geometry ();
    }

    /* Warp and save structure set (except dicom) */
    printf ("Rtds_warp: warp and save ss.\n");
    warp_and_save_ss (rtds, &xform, &pih, parms);

    /* Save dicom */
    if (bstring_not_empty (parms->output_dicom)) {
	printf ("Rtds_warp: Save dicom.\n");
	rtds->save_dicom ((const char*) parms->output_dicom);
    }
}
