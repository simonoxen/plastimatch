/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pointset_h_
#define _pointset_h_

#include "plm_config.h"
#include <string>
#include <vector>

class Labeled_point {
public:
    std::string label;
    float p[3];
};

typedef struct pointset Pointset;
struct pointset {
    int num_points;
    float *points;
};

class gpuit_EXPORT Pointset_new {
public:
    std::vector<Labeled_point> point_list;
public:
    void load_fcsv (const char *fn);
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
void
pointset_save_fcsv_by_cluster (Pointset* ps, int *clust_id, int which_cluster, const char *fn);
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
pointset_add_point (Pointset *ps, float lm[3]);
gpuit_EXPORT
void
pointset_add_point_noadjust (Pointset *ps, float lm[3]);
gpuit_EXPORT
void
pointset_debug (Pointset* ps);

#if defined __cplusplus
}
#endif

#endif
