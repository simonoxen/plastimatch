/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "cxt_to_mha.h"
#include "cxt_warp.h"
#include "file_util.h"
#include "gdcm_rtss.h"
#include "plm_file_format.h"
#include "plm_image_header.h"
#include "readmha.h"
#include "warp_parms.h"
#include "xform.h"

void
cxt_to_mha_write (Cxt_structure_list *structures, Warp_parms *parms)
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
	    //write_mha (fn, ctm_state.uchar_vol);
	    plm_image_save_vol (fn, ctm_state.uchar_vol);
	}
    }
    /* Write out labelmap, ss_img */
    if (parms->labelmap_fn[0]) {
	//write_mha (parms->labelmap_fn, ctm_state.labelmap_vol);
	plm_image_save_vol (parms->labelmap_fn, ctm_state.labelmap_vol);
    }
    if (parms->ss_img_fn[0]) {
	//write_mha (parms->ss_img_fn, ctm_state.ss_img_vol);
	plm_image_save_vol (parms->ss_img_fn, ctm_state.ss_img_vol);
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
