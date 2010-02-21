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
save_ss_img (Cxt_structure_list *cxt, Warp_parms *parms)
{
    Cxt_to_mha_state ctm_state;

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

    /* Write out labelmap, ss_img */
    if (parms->output_labelmap_fn[0]) {
	//write_mha (parms->labelmap_fn, ctm_state.labelmap_vol);
	plm_image_save_vol (parms->output_labelmap_fn, ctm_state.labelmap_vol);
    }
    if (parms->output_ss_img[0]) {
	//write_mha (parms->ss_img_fn, ctm_state.ss_img_vol);
	plm_image_save_vol (parms->output_ss_img, ctm_state.ss_img_vol);
    }

    /* Write out list of structure names */
    if (parms->output_ss_list[0]) {
	int i;
	FILE *fp;
	make_directory_recursive (parms->output_ss_list);
	fp = fopen (parms->output_ss_list, "w");
	for (i = 0; i < cxt->num_structures; i++) {
	    Cxt_structure *curr_structure;
	    curr_structure = &cxt->slist[i];
	    fprintf (fp, "%d|%s|%s\n",
		i, 
		(curr_structure->color 
		    ? (const char*) curr_structure->color->data 
		    : "\255\\0\\0"),
		curr_structure->name);
	}
	fclose (fp);
    }

    /* Free ctm_state */
    cxt_to_mha_free (&ctm_state);
}


static void
save_ss_output (Rtds *rtds,  Warp_parms *parms)
{
    if (!rtds->m_cxt) {
	return;
    }

#if defined (commentout)
    if (parms->output_img[0]) {
	/* If user didn't specify output format, see if we can guess from 
	   filename extension */
	if (parms->output_format == PLM_FILE_FMT_UNKNOWN) {
	    parms->output_format = plm_file_format_from_extension (
		parms->output_img);
	}

	/* Save output */
	switch (parms->output_format) {
	case PLM_FILE_FMT_CXT:
	    cxt_write (rtds->m_cxt, parms->output_img, true);
	    break;
	case PLM_FILE_FMT_DICOM_RTSS:
	case PLM_FILE_FMT_DICOM_DIR:
	    cxt_adjust_structure_names (rtds->m_cxt);
	    gdcm_rtss_save (rtds->m_cxt, parms->output_img, parms->dicom_dir);
	    break;
	case PLM_FILE_FMT_IMG:
	default:
	    cxt_to_mha_write (rtds->m_cxt, parms);
	    break;
	}
    }
#endif

    if (parms->output_labelmap_fn[0] || parms->output_ss_img[0]
	|| parms->output_ss_list[0] || parms->output_prefix[0])
    {
	save_ss_img (rtds->m_cxt, parms);
    }

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

    /* Do the warp */
    if (parms->xf_in_fn[0] && rtds->m_img) {
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
	//rtds->m_img->save_image (parms->output_img);
	rtds->m_img->convert_and_save (parms->output_img, parms->output_type);
    }

    /* Save output vector field */
    if (parms->xf_in_fn[0] && parms->output_vf[0]) {
	printf ("Saving vf...\n");
	itk_image_save (vf, parms->output_vf);
    }

    /* Save output structure set */
    save_ss_output (rtds, parms);
}
