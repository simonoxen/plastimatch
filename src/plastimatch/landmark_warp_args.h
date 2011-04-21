/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _landmark_warp_opts_h_
#define _landmark_warp_opts_h_

#include "plm_config.h"
#include "bstrwrap.h"

enum Landmark_warp_algorithm {
    LANDMARK_WARP_ALGORITHM_ITK_TPS,
    LANDMARK_WARP_ALGORITHM_RBF_GCS,
    LANDMARK_WARP_ALGORITHM_RBF_NSH
};

class Landmark_warp_args {
public:
    CBString input_fixed_landmarks_fn;
    CBString input_moving_landmarks_fn;
    CBString input_vf_fn;
    CBString input_xform_fn;
    CBString input_moving_image_fn;
    CBString output_warped_image_fn;
    CBString output_vf_fn;
    Landmark_warp_algorithm m_algorithm;
    float m_rbf_radius;
    float m_rbf_young_modulus;
public:
    Landmark_warp_args () {
        m_algorithm = LANDMARK_WARP_ALGORITHM_RBF_GCS;
	m_rbf_radius = 50.0f;           /* 5 cm default size */
	m_rbf_young_modulus = 0.0f;     /* default is no regularization */
    }
    void parse_args (int argc, char* argv[]);
};

void landmark_warp_opts_parse_args (Landmark_warp_args* options, int argc, char* argv[]);

#endif
