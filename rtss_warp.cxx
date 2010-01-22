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
#include "rtss_warp.h"
#include "warp_parms.h"
#include "xform.h"

void
rtss_warp (Warp_parms *parms)
{
    Cxt_structure_list structures;

    /* GCS FIX: Warping not implemented yet */
    if (parms->xf_in_fn[0]) {
	print_and_exit ("Sorry, plastimatch warp of dicom_rtss is not yet supported\n");
    }

    /* Load structures */
    cxt_initialize (&structures);
    gdcm_rtss_load (&structures, parms->input_fn, parms->dicom_dir);

#if defined (commentout)
    /* Set output size, resolution */
    if (parms->fixed_im_fn[0]) {
	FloatImageType::Pointer fixed = load_float (parms->fixed_im_fn, 0);
	PlmImageHeader pih;
	
	pih.set_from_itk_image (fixed);
	pih.get_gpuit_origin (structures.offset);
	pih.get_gpuit_spacing (structures.spacing);
	pih.get_gpuit_dim (structures.dim);
	structures.have_geometry = 1;

	cxt_apply_geometry (&structures);
    }
#endif
    /* GCS 2010/01/22: Change from itk_image to plm_image */
    /* Set output size, resolution */
    if (parms->fixed_im_fn[0]) {
	PlmImage pli;
	pli.load_native (parms->fixed_im_fn);
	cxt_set_geometry_from_plm_image (&structures, &pli);
    }

    cxt_debug (&structures);

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
	cxt_to_mha_write (&structures, parms);
	break;
    }
}
