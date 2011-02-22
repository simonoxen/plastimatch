/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rbf_cluster_h_
#define _rbf_cluster_h_

#include "plm_config.h"
#include "landmark_warp.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void rbf_cluster_kmeans_plusplus(Landmark_warp *lw);

plastimatch1_EXPORT
void rbf_cluster_find_adapt_radius(Landmark_warp *lw);

#if defined __cplusplus
}
#endif

#endif
