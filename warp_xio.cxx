/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <string>
#include "cxt_apply_dicom.h"
#include "cxt_io.h"
#include "plm_image.h"
#include "print_and_exit.h"
#include "warp_parms.h"
#include "warp_xio.h"
#include "xio_ct.h"
#include "xio_dir.h"
#include "xio_structures.h"

void
warp_xio_main (Warp_parms* parms)
{
    Cxt_structure_list cxt;
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

    /* Load structures from xio */
    //xio_structures_load (&cxt, xsd->path, parms->x_adj, parms->y_adj);
    xio_structures_load (&cxt, xsd->path, 0, 0);

    PlmImage pli;
    xio_ct_load (&pli, xsd->path);

    /* Set dicom uids, etc. */
    if (parms->dicom_dir[0]) {
	cxt_apply_dicom_dir (&cxt, parms->dicom_dir);
	//cxt.offset[0] += parms->x_adj;
	//cxt.offset[1] += parms->y_adj;
    }

    /* Write out the cxt */
    cxt_write (&cxt, parms->output_fn, true);
}
