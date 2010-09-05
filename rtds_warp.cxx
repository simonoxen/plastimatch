/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "bstring_util.h"
#include "cxt_apply_dicom.h"
#include "cxt_extract.h"
#include "cxt_to_mha.h"
#include "file_util.h"
#include "gdcm_dose.h"
#include "gdcm_rtss.h"
#include "plm_image_type.h"
#include "plm_warp.h"
#include "rtds_dicom.h"
#include "rtds_warp.h"
#include "ss_img_extract.h"
#include "xio_dose.h"
#include "xio_structures.h"

static void
compose_prefix_fn (
    CBString *fn, 
    const CBString *structure_name, 
    Warp_parms *parms
)
{
    fn->format ("%s_%s.%s", 
	(const char*) parms->output_prefix, 
	(const char*) (*structure_name), 
	"mha");
}

static void
prefix_output_save (Rtds *rtds, Warp_parms *parms)
{
    int i;
    Plm_image *ss_img;

    ss_img = rtds->m_ss_img;
    if (!ss_img) {
	return;
    }

    /* Use m_cxt or m_ss_list ?? */
    for (i = 0; i < rtds->m_cxt->num_structures; i++) {
	CBString fn;
	Cxt_structure *curr_structure = &rtds->m_cxt->slist[i];
	int bit = curr_structure->bit;

	if (bit == -1) continue;

	rtds->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
	UCharImageType::Pointer prefix_img = ss_img_extract (
	    rtds->m_ss_img->m_itk_uint32, bit);

	compose_prefix_fn (&fn, &curr_structure->name, parms);
	printf ("Trying to save prefix image: [%d,%d], %s\n", 
	    i, bit, (const char*) fn);
	itk_image_save (prefix_img, (const char*) fn);
    }
}

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
		&& bstring_not_empty (parms->dicom_dir))
	    {
		rtds_patient_pos_from_dicom_dir (rtds, parms->dicom_dir);
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
		(const char*) parms->dicom_dir, 
		parms->patient_pos);
	    break;
	case PLM_FILE_FMT_DIJ:
	    print_and_exit (
		"Warping dij files requires ctatts_in, dif_in files\n");
	    break;
	case PLM_FILE_FMT_DICOM_RTSS:
	    rtds->m_cxt = cxt_create ();
	    gdcm_rtss_load (
		rtds->m_cxt, 
		(const char*) parms->input_fn, 
		(const char*) parms->dicom_dir);
	    break;
	case PLM_FILE_FMT_DICOM_DOSE:
	    rtds->m_dose = gdcm_dose_load (
		0, 
		(const char*) parms->input_fn, 
		(const char*) parms->dicom_dir);
	    break;
	case PLM_FILE_FMT_CXT:
	    rtds->m_cxt = cxt_create ();
	    cxt_load (rtds->m_cxt, (const char*) parms->input_fn);
	    break;
	default:
	    print_and_exit (
		"Sorry, don't know how to convert/warp input type %s (%s)\n",
		plm_file_format_string (file_type),
		(const char*) parms->input_fn);
	    break;
	}
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
rasterize_ss_img (
    Rtds *rtds, 
    Xform *xf, 
    Plm_image_header *pih, 
    Warp_parms *parms)
{
    /* If we have need to create image outputs, or if we have to 
       warp something, then we need to rasterize the volume */
    /* GCS FIX: If there is an input m_ss_img, we still do this 
       because we might need the labelmap */
    if (bstring_not_empty (parms->output_labelmap_fn)
	|| bstring_not_empty (parms->output_ss_img_fn)
	|| bstring_not_empty (parms->xf_in_fn)
	|| bstring_not_empty (parms->output_prefix))
    {
	/* Rasterize structure sets */
	Cxt_to_mha_state *ctm_state;
	printf ("Rasterizing...\n");
	ctm_state = cxt_to_mha_create (rtds->m_cxt);

	/* Convert rasterized structure sets from vol to plm_image */
	printf ("Converting...\n");
	rtds->m_labelmap = new Plm_image;
	rtds->m_labelmap->set_gpuit (ctm_state->labelmap_vol);
	ctm_state->labelmap_vol = 0;
	if (rtds->m_ss_img) {
	    delete rtds->m_ss_img;
	}
	rtds->m_ss_img = new Plm_image;
	rtds->m_ss_img->set_gpuit (ctm_state->ss_img_vol);
	ctm_state->ss_img_vol = 0;

	/* We're done with cxt_state now */
	printf ("Destroying...\n");
	cxt_to_mha_destroy (ctm_state);
    }

    printf ("Finished rasterization.\n");

#if defined (commentout)
    /* GCS NOTE: This code will create prefix files "on-the-fly".  
       It is faster than what we do here, but can't work when we warp
       the images. */
    cxt_to_mha_init (&ctm_state, cxt, true, true, true);
    while (cxt_to_mha_process_next (&ctm_state, cxt)) {
	/* Write out prefix images */
	if (bstring_not_empty (&parms->output_prefix)) {
	    char fn[_MAX_PATH];
	    strcpy (fn, parms->output_prefix);
	    strcat (fn, "_");
	    strcat (fn, cxt_to_mha_current_name (&ctm_state, cxt));
	    strcat (fn, ".mha");
	    plm_image_save_vol (fn, ctm_state.uchar_vol);
	}
    }
#endif

}

static void
warp_ss_img (
    Rtds *rtds, 
    Xform *xf, 
    Plm_image_header *pih, 
    Warp_parms *parms)
{
    /* GCS FIX: This is inefficient.  We don't need to warp labelmap if 
       not included in output. */
    /* If we are warping, warp rasterized image(s) */
    if (bstring_not_empty (parms->xf_in_fn)) {
	Plm_image *tmp;

	tmp = new Plm_image;
	plm_warp (tmp, 0, xf, pih, rtds->m_labelmap, 0, parms->use_itk, 0);
	delete rtds->m_labelmap;
	rtds->m_labelmap = tmp;
	rtds->m_labelmap->convert (PLM_IMG_TYPE_ITK_ULONG);

	tmp = new Plm_image;
	plm_warp (tmp, 0, xf, pih, rtds->m_ss_img, 0, parms->use_itk, 0);
	delete rtds->m_ss_img;
	rtds->m_ss_img = tmp;
	rtds->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);

	/* The cxt is obsolete, but we can't delete it because it 
	   contains our "bits", used by prefix extraction.  */
    }
}

static void
save_ss_img (Rtds *rtds, Xform *xf, 
    Plm_image_header *pih, Warp_parms *parms)
{
    /* Write out labelmap, ss_img */
    if (bstring_not_empty (parms->output_labelmap_fn)) {
	printf ("Writing labelmap.\n");
	rtds->m_labelmap->save_image (parms->output_labelmap_fn);
	printf ("Done.\n");
    }
    if (bstring_not_empty (parms->output_ss_img_fn)) {
	printf ("Writing ss img.\n");
	rtds->m_ss_img->save_image ((const char*) parms->output_ss_img_fn);
	printf ("Done.\n");
    }

    /* Write out prefix images .. */
    if (bstring_not_empty (parms->output_prefix)) {
	printf ("Writing prefix images.\n");
	prefix_output_save (rtds, parms);
	printf ("Done.\n");
    }

    /* Write out list of structure names */
    if (bstring_not_empty (parms->output_ss_list_fn)) {
	printf ("Writing ss list.\n");
	int i;
	FILE *fp;
	make_directory_recursive ((const char*) parms->output_ss_list_fn);
	fp = fopen ((const char*) parms->output_ss_list_fn, "w");
	for (i = 0; i < rtds->m_cxt->num_structures; i++) {
	    Cxt_structure *curr_structure;
	    curr_structure = &rtds->m_cxt->slist[i];
	    fprintf (fp, "%d|%s|%s\n", 
		curr_structure->bit, 
		(bstring_empty (curr_structure->color) 
		    ? "255\\0\\0"
		    : (const char*) curr_structure->color),
		(const char*) curr_structure->name);
	}
	fclose (fp);
	printf ("Done.\n");
    }

    /* If we are warping, re-extract polylines into cxt */
    /* GCS FIX: This is only necessary if we are outputting polylines. 
       Otherwise it is  wasting users time. */
    if (bstring_not_empty (parms->xf_in_fn)) {
	printf ("Re-extracting cxt.\n");
	cxt_free_all_polylines (rtds->m_cxt);
	rtds->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
	cxt_extract (rtds->m_cxt, rtds->m_ss_img->m_itk_uint32, 
	    rtds->m_cxt->num_structures, true);
	printf ("Done.\n");
    }
}

static void
save_ss_output (
    Rtds *rtds,  
    Xform *xf, 
    Plm_image_header *pih, 
    Warp_parms *parms)
{
    if (!rtds->m_cxt) {
	return;
    }

    rasterize_ss_img (rtds, xf, pih, parms);

    warp_ss_img (rtds, xf, pih, parms);

    save_ss_img (rtds, xf, pih, parms);

    if (bstring_not_empty (parms->output_cxt_fn)) {
	cxt_save (rtds->m_cxt, (const char*) parms->output_cxt_fn, false);
    }

    if (bstring_not_empty (parms->output_xio_dirname)) {
	printf ("Saving xio format...\n");
	xio_structures_save (rtds->m_cxt, 
	    rtds->m_xio_transform,
	    parms->output_xio_version,
	    (const char*) parms->output_xio_dirname);
	printf ("Done.\n");
    }
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
    } else if (xform.m_type == XFORM_ITK_VECTOR_FIELD) {
	/* use the spacing from input vector field */
	pih.set_from_itk_image (xform.get_itk_vf());
    } else if (xform.m_type == XFORM_GPUIT_BSPLINE) {
	/* use the spacing from input bxf file */
	pih.set_from_gpuit_bspline (xform.get_gpuit_bsp());
    } else if (rtds->m_img) {
	/* use the spacing of the input image */
	pih.set_from_plm_image (rtds->m_img);
    } else if (rtds->m_ss_img) {
	/* use the spacing of the input image */
	pih.set_from_plm_image (rtds->m_ss_img);
    } else if (rtds->m_cxt) {
	/* use the spacing of the structure set */
	pih.set_from_gpuit (rtds->m_cxt->offset, rtds->m_cxt->spacing, 
	    rtds->m_cxt->dim, 0);
    } else if (rtds->m_dose) {
	/* use the spacing of dose */
	pih.set_from_plm_image (rtds->m_dose);
    } else {
	/* out of options?  :( */
	print_and_exit ("Sorry, I couldn't determine the output geometry\n");
    }

    printf ("PIH is:\n");
    pih.print ();

    /* Convert ss_img to cxt */
    if (rtds->m_ss_img) {
	rtds->convert_ss_img_to_cxt ();
    }

    /* Set UIDs, patient name, etc. */
    if (bstring_not_empty (parms->dicom_dir) && rtds->m_cxt) {
	cxt_apply_dicom_dir (rtds->m_cxt, (const char*) parms->dicom_dir);
    }

    /* Delete empty structures */
    if (parms->prune_empty && rtds->m_cxt) {
	cxt_prune_empty (rtds->m_cxt);
    }

    /* Set the geometry */
    if (rtds->m_cxt) {
	cxt_set_geometry_from_plm_image_header (rtds->m_cxt, &pih);
    }

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

    /* Save output dose image */
    if (bstring_not_empty (parms->output_dose_img_fn) && rtds->m_dose) {
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

    /* Save dose as ITK image */
    if (bstring_not_empty (parms->output_dose_img_fn) && rtds->m_img) {
	printf ("Saving dose image...\n");
	rtds->m_dose->convert_and_save (
	    (const char*) parms->output_dose_img_fn, 
	    parms->output_type);
    }

    /* Save output vector field */
    if (bstring_not_empty (parms->xf_in_fn) 
	&& bstring_not_empty (parms->output_vf_fn))
    {
	printf ("Saving vf...\n");
	itk_image_save (vf, (const char*) parms->output_vf_fn);
    }

    /* Warp and save output structure set */
    save_ss_output (rtds, &xform, &pih, parms);

    /* Save dicom */
    if (bstring_not_empty (parms->output_dicom)) {
	rtds->save_dicom ((const char*) parms->output_dicom);
    }
}
