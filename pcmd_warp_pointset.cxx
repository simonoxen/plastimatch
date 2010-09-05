/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "pcmd_warp.h"
#include "itk_pointset.h"
#include "xform.h"

void
warp_pointset_main (Warp_parms* parms)
{
    Xform xf;
    PointSetType::Pointer ps_in = PointSetType::New ();

    itk_pointset_load (ps_in, (const char*) parms->input_fn);
    itk_pointset_debug (ps_in);

    xform_load (&xf, parms->xf_in_fn);

    PointSetType::Pointer ps_out = itk_pointset_warp (ps_in, &xf);
    itk_pointset_debug (ps_out);
}
