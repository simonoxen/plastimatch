/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _raw_pointset_h_
#define _raw_pointset_h_

#include "plm_config.h"

typedef struct raw_pointset Raw_pointset;
struct raw_pointset {
    int num_points;
    float *points;
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
Raw_pointset*
pointset_load (const char *fn);
gpuit_EXPORT
void
pointset_save (Raw_pointset* ps, const char *fn);
gpuit_EXPORT
void
pointset_save_fcsv_by_cluster (Raw_pointset* ps, int *clust_id, int which_cluster, const char *fn);
gpuit_EXPORT
Raw_pointset *
pointset_create (void);
gpuit_EXPORT
void
pointset_destroy (Raw_pointset *ps);

gpuit_EXPORT
void
pointset_resize (Raw_pointset *ps, int new_size);
gpuit_EXPORT
void
pointset_add_point (Raw_pointset *ps, float lm[3]);
gpuit_EXPORT
void
pointset_add_point_noadjust (Raw_pointset *ps, float lm[3]);
gpuit_EXPORT
void
pointset_debug (Raw_pointset* ps);

#if defined __cplusplus
}
#endif

#endif
