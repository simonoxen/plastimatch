/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "cxt_to_mha.h"
#include "file_util.h"
#include "gdcm_rtss.h"
#include "plm_file_format.h"
#include "plm_image_header.h"
#include "readmha.h"
#include "warp_main.h"
#include "xform.h"


static void
do_cxt_to_mha_write (Cxt_structure_list *structures, Warp_parms *parms)
{
    Cxt_to_mha_state ctm_state;

    cxt_to_mha_init (&ctm_state, structures, true, true, true);

    while (cxt_to_mha_process_next (&ctm_state, structures)) {
	/* Write out prefix images */
	if (parms->prefix[0]) {
	    char fn[_MAX_PATH];
	    strcat (fn, parms->prefix);
	    strcat (fn, "_");
	    strcat (fn, cxt_to_mha_current_name (&ctm_state, structures));
	    strcat (fn, ".mha");
	    write_mha (fn, ctm_state.uchar_vol);
	}
    }
    /* Write out labelmap, ss_img */
    if (parms->labelmap_fn[0]) {
	write_mha (parms->labelmap_fn, ctm_state.labelmap_vol);
    }
    if (parms->output_fn[0]) {
	write_mha (parms->output_fn, ctm_state.ss_img_vol);
    }
    else if (parms->output_fn[0]) {
	write_mha (parms->ss_img_fn, ctm_state.ss_img_vol);
    }

    /* Write out list of structure names */
    if (parms->ss_list_fn[0]) {
	int i;
	FILE *fp;
	make_directory_recursive (parms->ss_list_fn);
	fp = fopen (parms->ss_list_fn, "w");
	for (i = 0; i < structures->num_structures; i++) {
	    Cxt_structure *curr_structure;
	    curr_structure = &structures->slist[i];
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

void
warp_dicom_rtss (Warp_parms *parms)
{
    Cxt_structure_list structures;

    /* GCS FIX: Warping not implemented yet */
    if (parms->xf_in_fn[0]) {
	print_and_exit ("Sorry, plastimatch warp of dicom_rtss is not yet supported\n");
    }

    /* Load structures */
    cxt_initialize (&structures);
    gdcm_rtss_load (&structures, parms->input_fn, parms->dicom_dir);

    /* Set output size, resolution */
    if (parms->fixed_im_fn[0]) {
	FloatImageType::Pointer fixed = load_float (parms->fixed_im_fn, 0);
	PlmImageHeader pih;
	
	pih.set_from_itk_image (fixed);
	pih.get_gpuit_origin (structures.offset);
	pih.get_gpuit_spacing (structures.spacing);
	pih.get_gpuit_dim (structures.dim);
    }

    /* If user didn't specify output format, see if we can guess from 
       filename extension */
    if (parms->output_format == PLM_FILE_FMT_UNKNOWN) {
	parms->output_format = plm_file_format_from_extension (
	    parms->output_fn);
    }

    /* Save output */
    switch (parms->output_format) {
    case PLM_FILE_FMT_CXT:
	cxt_write (&structures, parms->output_fn, true);
	break;
    case PLM_FILE_FMT_DICOM_RTSS:
    case PLM_FILE_FMT_DICOM_DIR:
	cxt_adjust_structure_names (&structures);
	gdcm_rtss_save (&structures, parms->output_fn, parms->dicom_dir);
	break;
    case PLM_FILE_FMT_IMG:
    default:
	do_cxt_to_mha_write (&structures, parms);
	break;
    }
}
