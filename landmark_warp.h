/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _landmark_warp_h_
#define _landmark_warp_h_

#include "plm_config.h"
#include "plm_image.h"
#include "pointset.h"

typedef struct landmark_warp Landmark_warp;
struct landmark_warp
{
    /* Inputs */
    Pointset *m_fixed_landmarks;
    Pointset *m_moving_landmarks;
    Plm_image *m_input_img;

    /* Config */
    float rbf_radius;
    float young_modulus;

    /* Outputs */
    bool m_want_warped_img;
    Plm_image *m_warped_img;
    bool m_want_vf;
    Plm_image *m_vf;
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT Landmark_warp*
landmark_warp_create (void);
gpuit_EXPORT void
landmark_warp_destroy (Landmark_warp *lw);
gpuit_EXPORT 
Landmark_warp*
landmark_warp_load_xform (const char *fn);
gpuit_EXPORT 
Landmark_warp*
landmark_warp_load_pointsets (const char *fixed_lm_fn, const char *moving_lm_fn);

#if defined __cplusplus
}
#endif

#endif
