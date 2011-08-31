/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "bstring_util.h"
#include "itk_pointset.h"
#include "pcmd_warp.h"
#include "xform.h"

void
warp_pointset_main (Warp_parms* parms)
{
    Xform xf;
    //FloatPointSetType::Pointer ps_in = FloatPointSetType::New ();

    //itk_pointset_load (ps_in, (const char*) parms->input_fn);
    //itk_pointset_debug (ps_in);

    Pointset_old *ps = pointset_load ((const char*) parms->input_fn);

    xform_load (&xf, parms->xf_in_fn);

    FloatPointSetType::Pointer itk_ps_in 
	= itk_float_pointset_from_pointset (ps);

    pointset_debug (ps);
    printf ("---\n");

    itk_pointset_debug (itk_ps_in);
    printf ("---\n");

    FloatPointSetType::Pointer itk_ps_out 
	= itk_pointset_warp (itk_ps_in, &xf);

    itk_pointset_debug (itk_ps_out);

    if (bstring_not_empty (parms->output_pointset_fn)) {
	Pointset_old *ps_out = pointset_from_itk_float_pointset (itk_ps_out);
	pointset_save (ps_out, (const char*) parms->output_pointset_fn);
	pointset_destroy (ps_out);
    }

    pointset_destroy (ps);
}
