/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <string>
#include "cxt_apply_dicom.h"
#include "cxt_io.h"
#include "cxt_warp.h"
#include "plm_image.h"
#include "print_and_exit.h"
#include "warp_parms.h"
#include "xio_ct.h"
#include "xio_dir.h"
#include "xio_structures.h"
#include "xio_warp.h"

void
xio_warp_main (Warp_parms* parms)
{
    //Cxt_structure_list cxt;
    Xio_dir *xd;
    Xio_patient_dir *xpd;
    Xio_studyset_dir *xsd;

    xd = xio_dir_create (parms->input_fn);

    if (xd->num_patient_dir <= 0) {
	print_and_exit ("Error, xio num_patient_dir = %d\n", 
	    xd->num_patient_dir);
    }
    xpd = &xd->patient_dir[0];
    if (xd->num_patient_dir > 1) {
	printf ("Warning: multiple patients found in xio directory.\n"
	    "Defaulting to first directory: %s\n", xpd->path);
    }

    if (xpd->num_studyset_dir <= 0) {
	print_and_exit ("Error, xio patient has no studyset.");
    }
    xsd = &xpd->studyset_dir[0];
    if (xpd->num_studyset_dir > 1) {
	printf ("Warning: multiple studyset found in xio patient directory.\n"
	    "Defaulting to first directory: %s\n", xsd->path);
    }

    PlmImage pli;

    /* Load the input image */
    xio_ct_load (&pli, xsd->path);


    /* Write out the image */
    if (parms->output_fn[0]) {
	pli.convert_and_save (parms->output_fn, PLM_IMG_TYPE_ITK_SHORT);
	printf ("Done writing xio ct.\n");
    }

#if defined (commentout)
    /* Set dicom uids, etc. */
    if (parms->dicom_dir[0]) {
	cxt_apply_dicom_dir (&cxt, parms->dicom_dir);
	//cxt.offset[0] += parms->x_adj;
	//cxt.offset[1] += parms->y_adj;
    }
#endif

    if (parms->ss_img_output_fn[0] 
	|| parms->labelmap_fn[0] 
	|| parms->prefix[0] 
	|| parms->cxt_output_fn[0] 
	|| parms->xio_output_dirname[0])
    {
	Cxt_structure_list cxt;

	/* Load structures from xio */
	printf ("calling xio_structures_load\n");
	xio_structures_load (&cxt, xsd->path, 0, 0);

	if (parms->prune_empty) {
	    cxt_prune_empty (&cxt);
	}

	/* Copy geometry from Xio CT to structures */
	printf ("calling cxt_set_geometry_from_plm_image\n");
	cxt_set_geometry_from_plm_image (&cxt, &pli);

	/* Write cxt output */
	if (parms->cxt_output_fn[0]) {
	    cxt_write (&cxt, parms->cxt_output_fn, 0);
	}

	/* Write xio output */
	if (parms->xio_output_dirname[0]) {
	    xio_structures_save (&cxt, parms->xio_output_dirname);
	}

	/* Convert and write output */
	printf ("calling cxt_to_mha_write\n");
	cxt_to_mha_write (&cxt, parms);
	printf ("done.\n");
    }
}
