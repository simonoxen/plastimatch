/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <string>
#include "print_and_exit.h"
#include "warp_parms.h"
#include "warp_xio.h"
#include "xio_dir.h"

void
warp_xio_main (Warp_parms* parms)
{
    Xio_dir *xd;
    Xio_patient_dir *xpd;
    std::string ct_path;

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

    switch (xpd->type) {
    case XPD_TOPLEVEL_PATIENT_DIR:
	/* GCS FIX: Need xio_dir to figure out studyset subdirectory */
	ct_path = std::string(xpd->path) + "/anatomy/studyset";
	break;
    case XPD_STUDYSET_DIR:
	ct_path = xpd->path;
	break;
    }
}
