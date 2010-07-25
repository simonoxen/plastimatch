/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "landmark_warp.h"
#include "logfile.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "vf.h"
#include "volume.h"

Landmark_warp*
landmark_warp_create (void)
{
    Landmark_warp *lw;
    lw = (Landmark_warp*) malloc (sizeof (Landmark_warp));
    memset (lw, 0, sizeof (Landmark_warp));
    return lw;
}

void
landmark_warp_destroy (Landmark_warp *lw)
{
    if (lw->moving) {
	pointset_destroy (lw->moving);
    }
    if (lw->fixed) {
	pointset_destroy (lw->fixed);
    }
    free (lw);
}

/* GCS FIX: Oops.  This doesn't work because tps_xform is c++ code.
   If needed, we need to separate out Tps_xform as a separate c file. */
Landmark_warp*
landmark_warp_load_xform (char *fn)
{
#if defined (commentout)
    Landmark_warp *lw;
    Tps_xform *tps;
    int i;

    tps = tps_xform_load (options->input_xform_fn);
    if (!tps) return 0;

    if (tps->num_tps_nodes <= 0) {
	tps_xform_destroy (tps);
	return 0;
    }

    lw = landmark_warp_create ();
    lw->fixed = pointset_create ();
    pointset_resize (lw->fixed, tps->num_tps_nodes);
    lw->moving = pointset_create ();
    pointset_resize (lw->moving, tps->num_tps_nodes);

    for (i = 0; i < tps->num_tps_nodes; i++) {
	lw->fixed[i*3 + 0] = tps->src[0];
	lw->fixed[i*3 + 1] = tps->src[1];
	lw->fixed[i*3 + 2] = tps->src[2];
	lw->moving[i*3 + 0] = tps->tgt[0];
	lw->moving[i*3 + 1] = tps->tgt[1];
	lw->moving[i*3 + 2] = tps->tgt[2];
    }

    /* Discard alpha values and image header. */
#endif

    return 0;
}

Landmark_warp*
landmark_warp_load_pointsets (char *fixed_lm_fn, char *moving_lm_fn)
{
    Landmark_warp *lw;

    lw = landmark_warp_create ();
    lw->fixed = pointset_load (fixed_lm_fn);
    lw->moving = pointset_load (moving_lm_fn);

    if (!lw->fixed || !lw->moving) {
	landmark_warp_destroy (lw);
	return 0;
    }
    return lw;
}
