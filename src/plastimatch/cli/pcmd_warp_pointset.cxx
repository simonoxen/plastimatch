/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "plmbase.h"

#include "itk_pointset.h"
#include "pcmd_warp.h"
#include "pstring.h"

void
warp_pointset_main (Warp_parms* parms)
{
    Xform xf;
    //FloatPointSetType::Pointer ps_in = FloatPointSetType::New ();

    //itk_pointset_load (ps_in, (const char*) parms->input_fn);
    //itk_pointset_debug (ps_in);

    //Raw_pointset *ps = pointset_load ((const char*) parms->input_fn);
    Unlabeled_pointset ps;
    ps.load ((const char*) parms->input_fn);

    xform_load (&xf, parms->xf_in_fn);

    //FloatPointSetType::Pointer itk_ps_in 
    //= itk_float_pointset_from_raw_pointset (ps);

    FloatPointSetType::Pointer itk_ps_in 
	= itk_float_pointset_from_pointset (&ps);

    //pointset_debug (ps);
    //printf ("---\n");

    //itk_pointset_debug (itk_ps_in);
    //printf ("---\n");

    FloatPointSetType::Pointer itk_ps_out 
	= itk_pointset_warp (itk_ps_in, &xf);

    //itk_pointset_debug (itk_ps_out);

    if (parms->output_pointset_fn.not_empty()) {
	//Raw_pointset *ps_out = raw_pointset_from_itk_float_pointset (itk_ps_out);
	//pointset_save (ps_out, (const char*) parms->output_pointset_fn);
	//pointset_destroy (ps_out);

	Unlabeled_pointset *ps_out =
	    unlabeled_pointset_from_itk_float_pointset (itk_ps_out);
	ps_out->save ((const char*) parms->output_pointset_fn);
	delete ps_out;
    }
}
