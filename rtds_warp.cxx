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
#include "plm_image_type.h"
#include "plm_warp.h"
#include "referenced_dicom_dir.h"
#include "rtds_dicom.h"
#include "rtds_warp.h"
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
	    /* GCS FIX: This is where the dicom directory gets loaded twice. 
	       We should (probably) remove this load */
	    rtds->m_ss_image->load_gdcm_rtss (
		(const char*) parms->input_fn, 
		(const char*) parms->referenced_dicom_dir);
	    break;
	case PLM_FILE_FMT_DICOM_DOSE:
	    rtds->m_dose = gdcm_dose_load (
		0, 
		(const char*) parms->input_fn, 
		(const char*) parms->referenced_dicom_dir);
	    break;
	case PLM_FILE_FMT_CXT:
	    rtds->m_ss_image = new Ss_image;
	    rtds->m_ss_image->load_cxt (parms->input_fn);
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
	printf ("Loading RDD\n");
	rtds->load_rdd ((const char*) parms->referenced_dicom_dir);
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
	rtds->load_dose_xio ((const char*) parms->input_dose_xio_fn);
    }

    if (bstring_not_empty (parms->input_dose_ast_fn)) {
	rtds->load_dose_astroid ((const char*) parms->input_dose_ast_fn);
    }

    if (bstring_not_empty (parms->input_dose_mc_fn)) {
	rtds->load_dose_mc ((const char*) parms->input_dose_mc_fn);
    }
}

static void
save_ss_img (Rtds *rtds, Xform *xf, 
    Plm_image_header *pih, Warp_parms *parms)
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
	rtds->m_ss_image->save_cxt (parms->output_cxt_fn, false);
    }

    /* xio */
    if (bstring_not_empty (parms->output_xio_dirname)) {
	printf ("Output xio dirname = %s\n", 
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
	rtds->m_ss_image->rasterize ();
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
	printf ("Loading xform (%s)\n", (const char*) parms->xf_in_fn);
	xform_load (&xform, (const char*) parms->xf_in_fn);
    }

    /* Try to guess the proper dimensions and spacing for output image */
    if (bstring_not_empty (parms->fixed_im_fn)) {
	/* use the spacing of user-supplied fixed image */
	FloatImageType::Pointer fixed = itk_image_load_float (
	    parms->fixed_im_fn, 0);
	pih.set_from_itk_image (fixed);
    } else if (rtds->m_rdd && rtds->m_rdd->m_loaded) {
	/* use spacing from referenced CT */
	printf ("Setting PIH from RDD\n");
	Plm_image_header::clone (&pih, &rtds->m_rdd->m_pih);
    } else if (xform.m_type == XFORM_ITK_VECTOR_FIELD) {
	/* use the spacing from input vector field */
	pih.set_from_itk_image (xform.get_itk_vf());
    } else if (xform.m_type == XFORM_GPUIT_BSPLINE) {
	/* use the spacing from input bxf file */
	pih.set_from_gpuit_bspline (xform.get_gpuit_bsp());
    } else if (rtds->m_img) {
	/* use the spacing of the input image */
	pih.set_from_plm_image (rtds->m_img);
    } else if (rtds->m_ss_image) {
	/* use the spacing of the input image */
	if (rtds->m_ss_image->m_ss_img) {
	    pih.set_from_plm_image (rtds->m_ss_image->m_ss_img);
	}
	/* use the spacing of the structure set */
	else if (rtds->m_ss_image->m_cxt) {
	    pih.set_from_gpuit (
		rtds->m_ss_image->m_cxt->offset, 
		rtds->m_ss_image->m_cxt->spacing, 
		rtds->m_ss_image->m_cxt->dim, 0);
	}
    } else if (rtds->m_dose) {
	/* use the spacing of dose */
	pih.set_from_plm_image (rtds->m_dose);
    } else {
	/* out of options?  :( */
	print_and_exit ("Sorry, I couldn't determine the output geometry\n");
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
	plm_warp (im_out, &vf, &xform, &pih, rtds->m_img, parms->default_val, 
	    parms->use_itk, parms->interp_lin);
	delete rtds->m_img;
	rtds->m_img = im_out;
    }

    /* Save output image */
    if (bstring_not_empty (parms->output_img_fn) && rtds->m_img) {
	printf ("Saving image...\n");
	rtds->m_img->convert_and_save (
	    (const char*) parms->output_img_fn, 
	    parms->output_type);
    }

    /* Warp the dose image */
    if (rtds->m_dose
	&& bstring_not_empty (parms->xf_in_fn)
	&& (bstring_not_empty (parms->output_dose_img_fn)
	    || bstring_not_empty (parms->output_xio_dirname)))
    {
	printf ("Warping dose image...\n");
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
	printf ("Saving dose image...\n");
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

	printf ("Saving xio dose...\n");
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
	printf ("Saving vf...\n");
	itk_image_save (vf, (const char*) parms->output_vf_fn);
    }

    /* Preprocess structure sets */
    if (rtds->m_ss_image) {

	/* Convert ss_img to cxt */
	rtds->m_ss_image->convert_ss_img_to_cxt ();

	/* Delete empty structures */
	if (parms->prune_empty) {
	    rtds->m_ss_image->prune_empty ();
	}

	/* Set the DICOM reference info */
	rtds->m_ss_image->apply_dicom_dir (rtds->m_rdd);

	/* Set the geometry */
	rtds->m_ss_image->set_geometry_from_plm_image_header (&pih);
    }

    /* Warp and save structure set (except dicom) */
    warp_and_save_ss (rtds, &xform, &pih, parms);

#if defined (commentout)
#endif
    /* In certain cases, we have to delay setting dicom uids 
       (e.g. wait until after warping) */
    /* GCS FIX: Sometimes referenced_dicom_dir is applied multiple times, 
       such as when using dicom and xio input, which is inefficient. */
    /* GCS: Which cases are these?  (It does seem to solve problems...) */
    if (rtds->m_ss_image && rtds->m_rdd) {
	rtds->m_ss_image->apply_dicom_dir (rtds->m_rdd);
    }

    /* Save dicom */
    if (bstring_not_empty (parms->output_dicom)) {
	rtds->save_dicom ((const char*) parms->output_dicom);
    }
}
