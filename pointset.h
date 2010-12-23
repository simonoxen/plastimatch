/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pointset_h_
#define _pointset_h_

#include "plm_config.h"

typedef struct pointset Pointset;
struct pointset {
    int num_points;
    float *points;
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
Pointset*
pointset_load (const char *fn);
gpuit_EXPORT
void
pointset_save (Pointset* ps, const char *fn);
gpuit_EXPORT
Pointset *
pointset_create (void);
gpuit_EXPORT
void
pointset_destroy (Pointset *ps);

gpuit_EXPORT
void
pointset_resize (Pointset *ps, int new_size);
gpuit_EXPORT
void
pointset_debug (Pointset* ps);

#if defined __cplusplus
}
#endif

#endif
