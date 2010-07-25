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
pointset_load (char *fn);
gpuit_EXPORT
void
pointset_destroy (Pointset *ps);

#if defined __cplusplus
}
#endif

#endif
