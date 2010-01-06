/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "warp_main.h"
#include "gdcm_rtss.h"
#include "xform.h"

void
warp_dicom_rtss (Warp_parms *parms)
{
    Cxt_structure_list structures;

    printf ("Running warp_dicom_rtss.\n");
    cxt_initialize (&structures);
    gdcm_rtss_load (&structures, parms->input_fn, parms->dicom_dir);

    printf ("Saving output.\n");
    cxt_write (&structures, parms->output_fn, true);
}
