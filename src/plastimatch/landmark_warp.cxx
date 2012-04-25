/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "plmbase.h"

#include "landmark_warp.h"
#include "logfile.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "vf.h"
#include "volume.h"

Landmark_warp::Landmark_warp (void)
{
    m_fixed_landmarks = 0;
    m_moving_landmarks = 0;
    m_input_img = 0;

    default_val = 0;
    rbf_radius = 0;
    young_modulus = 0;
    num_clusters = 0;

    cluster_id = 0;
    adapt_radius = 0;

    m_warped_img = 0;
    m_vf = 0;
    m_warped_landmarks = 0;
}

Landmark_warp::~Landmark_warp (void)
{
    if (m_moving_landmarks) {
	pointset_destroy (m_moving_landmarks);
    }
    if (m_fixed_landmarks) {
	pointset_destroy (m_fixed_landmarks);
    }
    if (m_warped_landmarks) {
	pointset_destroy (m_warped_landmarks);
    }
    if (cluster_id) free(cluster_id);
    if (adapt_radius) free(adapt_radius);
}

Landmark_warp*
landmark_warp_create (void)
{
    return new Landmark_warp;
}

void
landmark_warp_destroy (Landmark_warp *lw)
{
    delete lw;
}

/* GCS FIX: Oops.  This doesn't work because tps_xform is c++ code.
   If needed, we need to separate out Tps_xform as a separate c file. */
Landmark_warp*
landmark_warp_load_xform (const char *fn)
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

void
Landmark_warp::load_pointsets (
    const char *fixed_lm_fn, 
    const char *moving_lm_fn
)
{
    m_fixed_landmarks = pointset_load (fixed_lm_fn);
    m_moving_landmarks = pointset_load (moving_lm_fn);
}

Landmark_warp*
landmark_warp_load_pointsets (const char *fixed_lm_fn, const char *moving_lm_fn)
{
    Landmark_warp *lw;

    lw = landmark_warp_create ();
    lw->load_pointsets (fixed_lm_fn, moving_lm_fn);

    if (!lw->m_fixed_landmarks || !lw->m_moving_landmarks) {
	landmark_warp_destroy (lw);
	return 0;
    }
    return lw;
}
