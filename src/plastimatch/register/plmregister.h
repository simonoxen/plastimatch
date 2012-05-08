/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plmregister_h_
#define _plmregister_h_

#include "plmregister_config.h"

#include "bspline.h"
#include "bspline_landmarks.h"
#include "bspline_mi.h"
#include "bspline_mse.h"
#include "bspline_optimize.h"
#include "bspline_optimize_lbfgsb.h"
#include "bspline_optimize_liblbfgs.h"
#if (NLOPT_FOUND)
#include "bspline_optimize_nlopt.h"
#endif
#include "bspline_optimize_steepest.h"
#include "bspline_regularize.h"
#include "bspline_regularize_analytic.h"
#include "bspline_regularize_numeric.h"
#include "demons.h"
#include "demons_misc.h"
#include "demons_state.h"
#if (!PLM_CUDA_COMPILE)
#include "landmark_warp.h"
#endif
#include "plm_parms.h"
#include "plm_stages.h"
#include "rbf_cluster.h"
#include "rbf_gauss.h"
#include "rbf_wendland.h"
#if (!PLM_CUDA_COMPILE)
#include "registration_data.h"
#endif

#endif
