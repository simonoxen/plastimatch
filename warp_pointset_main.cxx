/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "warp_main.h"
#include "itk_pointset.h"
#include "xform.h"

void
warp_pointset_main (Warp_Parms* parms)
{
    Xform xf;
    PointSetType::Pointer ps_in = PointSetType::New ();

    pointset_load (ps_in, parms->mha_in_fn);
    pointset_debug (ps_in);

    load_xform (&xf, parms->xf_in_fn);

    PointSetType::Pointer ps_out = pointset_warp (ps_in, &xf);
    pointset_debug (ps_out);
}

