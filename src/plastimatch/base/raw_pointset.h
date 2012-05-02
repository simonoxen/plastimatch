/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _raw_pointset_h_
#define _raw_pointset_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmbase.h"
 */

#include "plmbase_config.h"

typedef struct raw_pointset Raw_pointset;
struct raw_pointset {
    int num_points;
    float *points;
};

C_API void pointset_add_point (
        Raw_pointset *ps,
        float lm[3]
);
C_API void pointset_add_point_noadjust (
        Raw_pointset *ps,
        float lm[3]
);
C_API Raw_pointset *pointset_create (void);
C_API void pointset_debug (Raw_pointset* ps);
C_API void pointset_destroy (Raw_pointset *ps);
C_API Raw_pointset* pointset_load (const char *fn);
C_API void pointset_resize (
        Raw_pointset *ps,
        int new_size
);
C_API void pointset_save (
        Raw_pointset* ps,
        const char *fn
);
C_API void pointset_save_fcsv_by_cluster (
        Raw_pointset* ps,
        int *clust_id,
        int which_cluster,
        const char *fn
);

#endif
