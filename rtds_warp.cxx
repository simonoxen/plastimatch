/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "cxt_apply_dicom.h"
#include "cxt_extract.h"
#include "cxt_to_mha.h"
#include "file_util.h"
#include "gdcm_rtss.h"
#include "plm_warp.h"
#include "rtds_warp.h"
#include "xio_structures.h"

static void
load_ss_img (Rtds *rtds, Warp_parms *parms)
{
    PlmImage *pli;
    int num_structs = -1;

    /* Load ss_img */
    printf ("Loading input file...\n");
    pli = plm_image_load_native (parms->input_ss_img);
    printf ("Done.\n");

    /* Allocate memory for cxt */
    rtds->m_cxt = cxt_create ();

    /* Set structure names */
    if (parms->input_ss_list[0]) {
	cxt_xorlist_read (rtds->m_cxt, parms->input_ss_list);
	num_structs = rtds->m_cxt->num_structures;
    }

    /* Copy geometry to cxt */
    cxt_set_geometry_from_plm_image (rtds->m_cxt, pli);

    /* Extract polylines */
    printf ("Running marching squares (%d structs)...\n", num_structs);
    pli->convert (PLM_IMG_TYPE_ITK_ULONG);
    cxt_extract (rtds->m_cxt, pli->m_itk_uint32, num_structs);
    printf ("Done.\n");

    /* Set UIDs */
    if (parms->dicom_dir[0]) {
	printf ("Parsing dicom...\n");
	cxt_apply_dicom_dir (rtds->m_cxt, parms->dicom_dir);
	printf ("Done.\n");
    }

    /* Free ss_img */
    delete pli;
}

static void
load_input_files (Rtds *rtds, Plm_file_format file_type, Warp_parms *parms)
{
    if (parms->input_fn[0]) {
	switch (file_type) {
	case PLM_FILE_FMT_NO_FILE:
	    print_and_exit ("Could not open input file %s for read\n",
		parms->input_fn);
	    break;
	case PLM_FILE_FMT_UNKNOWN:
	case PLM_FILE_FMT_IMG:
	    rtds->m_img = plm_image_load_native (parms->input_fn);
	    //warp_image_main (&parms);
	    break;
	case PLM_FILE_FMT_DICOM_DIR:
	    /* GCS FIX: Need to load rtss too */
	    rtds->m_img = plm_image_load_native (parms->input_fn);
	    //warp_image_main (&parms);
	    break;
	case PLM_FILE_FMT_XIO_DIR:
	    rtds->load_xio (parms->input_fn);
	    //xio_warp_main (&parms);
	    break;
	case PLM_FILE_FMT_DIJ:
	    print_and_exit ("Warping dij files requires ctatts_in, dif_in files\n");
	    break;
	case PLM_FILE_FMT_DICOM_RTSS:
	    rtds->m_cxt = cxt_create ();
	    gdcm_rtss_load (rtds->m_cxt, parms->input_fn, parms->dicom_dir);
	    //rtss_warp (&parms);
	    break;
	case PLM_FILE_FMT_CXT:
	    rtds->m_cxt = cxt_create ();
	    cxt_read (rtds->m_cxt, parms->input_fn);
	    //ctx_warp (&parms);
	    break;
	default:
	    print_and_exit (
		"Sorry, don't know how to convert/warp input type %s (%s)\n",
		plm_file_format_string (file_type),
		parms->input_fn);
	    break;
	}
    }

    if (parms->input_ss_img[0]) {
	load_ss_img (rtds, parms);
    }
}

static void
warp_and_save_ss_img (Rtds *rtds, Xform *xf, 
    PlmImageHeader *pih, Warp_parms *parms)
{
    PlmImage *pli_labelmap = new PlmImage;
    PlmImage *pli_ss_img = new PlmImage;

    printf ("Saving ss_img...\n");

    /* If we have need to create image outputs, or if we have to 
       warp something, then we need to rasterize the volume */
    if (parms->output_labelmap_fn[0] || parms->output_ss_img[0]
	|| parms->xf_in_fn[0] || parms->output_prefix[0])
    {
	/* Rasterize structure sets */
	Cxt_to_mha_state *ctm_state;
	ctm_state = cxt_to_mha_create (rtds->m_cxt);

	/* Convert rasterized structure sets from vol to plm_image */
	pli_labelmap->set_gpuit (ctm_state->labelmap_vol);
	ctm_state->labelmap_vol = 0;
	pli_ss_img->set_gpuit (ctm_state->ss_img_vol);
	ctm_state->ss_img_vol = 0;

	/* We're done with cxt_state now */
	cxt_to_mha_destroy (ctm_state);
    }

#if defined (commentout)
    /* GCS NOTE: This code will create prefix files "on-the-fly".  
       It is faster than what we do here, but can't work when we warp
       the images. */
    cxt_to_mha_init (&ctm_state, cxt, true, true, true);
    while (cxt_to_mha_process_next (&ctm_state, cxt)) {
	/* Write out prefix images */
	if (parms->output_prefix[0]) {
	    char fn[_MAX_PATH];
	    strcpy (fn, parms->output_prefix);
	    strcat (fn, "_");
	    strcat (fn, cxt_to_mha_current_name (&ctm_state, cxt));
	    strcat (fn, ".mha");
	    plm_image_save_vol (fn, ctm_state.uchar_vol);
	}
    }
#endif

    /* If we are warping, warp rasterized image(s) */
    if (parms->xf_in_fn[0]) {
	PlmImage *tmp;

	tmp = new PlmImage;
	plm_warp (tmp, 0, xf, pih, pli_labelmap, 0, parms->use_itk, 0);
	delete pli_labelmap;
	pli_labelmap = tmp;

	tmp = new PlmImage;
	plm_warp (tmp, 0, xf, pih, pli_ss_img, 0, parms->use_itk, 0);
	delete pli_ss_img;
	pli_ss_img = tmp;
    }

    /* Write out labelmap, ss_img */
    if (parms->output_labelmap_fn[0]) {
	printf ("Writing labelmap.\n");
	pli_labelmap->save_image (parms->output_labelmap_fn);
	printf ("Done.\n");
    }
    if (parms->output_ss_img[0]) {
	printf ("Writing ss img.\n");
	pli_ss_img->save_image (parms->output_ss_img);
	printf ("Done.\n");
    }

    /* Write out list of structure names */
    if (parms->output_ss_list[0]) {
	int i;
	FILE *fp;
	make_directory_recursive (parms->output_ss_list);
	fp = fopen (parms->output_ss_list, "w");
	for (i = 0; i < rtds->m_cxt->num_structures; i++) {
	    Cxt_structure *curr_structure;
	    curr_structure = &rtds->m_cxt->slist[i];
	    fprintf (fp, "%d|%s|%s\n",
		i, 
		(curr_structure->color 
		    ? (const char*) curr_structure->color->data 
		    : "\255\\0\\0"),
		curr_structure->name);
	}
	fclose (fp);
    }

    /* Write out prefix images .. */
    if (parms->output_prefix[0]) {
	/* GCS FIX */
	print_and_exit ("Sorry, prefix image export disabled.\n");
    }

    /* If we are warping, re-extract polylines into cxt */
    if (parms->xf_in_fn[0]) {
	cxt_free_all_polylines (rtds->m_cxt);
	cxt_extract (rtds->m_cxt, pli_ss_img->m_itk_uint32, 
	    rtds->m_cxt->num_structures);
    }
}

static void
save_ss_output (Rtds *rtds,  Xform *xf, 
    PlmImageHeader *pih, Warp_parms *parms)
{
    if (!rtds->m_cxt) {
	return;
    }

    warp_and_save_ss_img (rtds, xf, pih, parms);

    if (parms->output_cxt[0]) {
	cxt_write (rtds->m_cxt, parms->output_cxt, true);
    }

    if (parms->output_xio_dirname[0]) {
	printf ("Saving xio format...\n");
	xio_structures_save (rtds->m_cxt, 
	    parms->output_xio_version, 
	    parms->output_xio_dirname);
	printf ("Done.\n");
    }

    if (parms->output_dicom[0]) {
	cxt_adjust_structure_names (rtds->m_cxt);
	gdcm_rtss_save (rtds->m_cxt, parms->output_dicom, parms->dicom_dir);
    }
}

void
rtds_warp (Rtds *rtds, Plm_file_format file_type, Warp_parms *parms)
{
    DeformationFieldType::Pointer vf = DeformationFieldType::New();
    Xform xform;
    PlmImageHeader pih;

    /* Load input file(s) */
    load_input_files (rtds, file_type, parms);

    /* Load transform */
    if (parms->xf_in_fn[0]) {
	printf ("Loading xform (%s)\n", parms->xf_in_fn);
	load_xform (&xform, parms->xf_in_fn);
    }

    /* Try to guess the proper dimensions and spacing for output image */
    if (parms->fixed_im_fn[0]) {
	/* use the spacing of user-supplied fixed image */
	FloatImageType::Pointer fixed = load_float (parms->fixed_im_fn, 0);
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
    } else if (rtds->m_cxt) {
	/* use the spacing of the structure set */
	pih.set_from_gpuit (rtds->m_cxt->offset, rtds->m_cxt->spacing, 
	    rtds->m_cxt->dim, 0);
    } else {
	/* out of options?  :( */
	print_and_exit ("Sorry, I couldn't determine the output geometry\n");
    }

    printf ("PIH is:\n");
    pih.print ();

    /* Set the geometry */
    if (rtds->m_cxt) {
	cxt_set_geometry_from_plm_image_header (rtds->m_cxt, &pih);
    }

    /* Warp the image and create vf */
    if (rtds->m_img && parms->xf_in_fn[0] 
	&& (parms->output_img[0] || parms->output_vf[0]))
    {
	PlmImage *im_out;
	im_out = new PlmImage;
	plm_warp (im_out, &vf, &xform, &pih, rtds->m_img, parms->default_val, 
	    parms->use_itk, parms->interp_lin);
	delete rtds->m_img;
	rtds->m_img = im_out;
    }

    /* Save output image */
    if (parms->output_img[0] && rtds->m_img) {
	printf ("Saving image...\n");
	rtds->m_img->convert_and_save (parms->output_img, parms->output_type);
    }

    /* Save output vector field */
    if (parms->xf_in_fn[0] && parms->output_vf[0]) {
	printf ("Saving vf...\n");
	itk_image_save (vf, parms->output_vf);
    }

    /* Warp and save output structure set */
    save_ss_output (rtds, &xform, &pih, parms);
}
