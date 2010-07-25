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
    Pointset *fixed;
    Pointset *moving;
    Plm_image *m_img;
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
landmark_warp_load_xform (char *fn);
gpuit_EXPORT 
Landmark_warp*
landmark_warp_load_pointsets (char *fixed_lm_fn, char *moving_lm_fn);

#if defined __cplusplus
}
#endif

#endif
