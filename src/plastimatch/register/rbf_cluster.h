/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rbf_cluster_h_
#define _rbf_cluster_h_

#include "plmregister_config.h"

class Landmark_warp;

API void rbf_cluster_kmeans_plusplus (Landmark_warp *lw);
API void rbf_cluster_find_adapt_radius (Landmark_warp *lw);

#endif
