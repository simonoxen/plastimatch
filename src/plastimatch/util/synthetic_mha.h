/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _synthetic_mha_h_
#define _synthetic_mha_h_

#include "plmutil_config.h"
#include <string>
#include "direction_cosines.h"
#include "plm_image_type.h"
#include "pstring.h"

class Rt_study;

enum Pattern_type {
    PATTERN_DOSE,
    PATTERN_GAUSS,
    PATTERN_RECT,
    PATTERN_SPHERE,
    PATTERN_MULTI_SPHERE,
    PATTERN_DONUT,
    PATTERN_GRID,
    PATTERN_LUNG,
    PATTERN_XRAMP,
    PATTERN_YRAMP,
    PATTERN_ZRAMP
};

class Synthetic_mha_parms {
public:
    int output_type;
    Pattern_type pattern;
    Pstring fixed_fn;
    std::string input_fn;
    int dim[3];
    float origin[3];
    float spacing[3];
    Direction_cosines dc;

    float background;
    float foreground;
    float background_alpha;
    float foreground_alpha;
    bool m_want_ss_img;
    bool m_want_dose_img;

    float gauss_center[3];
    float gauss_std[3];
    float penumbra;
    float dose_size[6];
    float dose_center[3];
    float rect_size[6];
    float sphere_center[3];
    float sphere_radius[3];
    float donut_center[3];
    float donut_radius[3];
    int donut_rings;
    int grid_spacing[3];
    float lung_tumor_pos[3];
    
    int num_multi_sphere;

public:
    Synthetic_mha_parms () {
        output_type = PLM_IMG_TYPE_ITK_FLOAT;
        pattern = PATTERN_GAUSS;
        input_fn = "";

        for (int i = 0; i < 3; i++) {
            spacing[i] = 5.0f;
            dim[i] = 100;
            origin[i] = 0.0f;
            gauss_center[i] = 0.0f;
            gauss_std[i] = 100.0f;
            sphere_center[i] = 0.0f;
            sphere_radius[i] = 50.0f;
            donut_center[i] = 0.0f;
            lung_tumor_pos[i] = 0.0f;
            dose_center[i] = 0.0f;
        }
        background = -1000.0f;
        foreground = 0.0f;
        background_alpha = 1.0f;
        foreground_alpha = 1.0f;
        m_want_ss_img = false;
        m_want_dose_img = false;
        rect_size[0] = -50.0f;
        rect_size[1] = +50.0f;
        rect_size[2] = -50.0f;
        rect_size[3] = +50.0f;
        rect_size[4] = -50.0f;
        rect_size[5] = +50.0f;
        donut_radius[0] = 50.0f;
        donut_radius[1] = 50.0f;
        donut_radius[2] = 20.0f;
        donut_rings = 2;
        grid_spacing[0] = 10;
        grid_spacing[1] = 10;
        grid_spacing[2] = 10;
        penumbra = 5.0f;
        dose_size[0] = -50.0f;
        dose_size[1] = +50.0f;
        dose_size[2] = -50.0f;
        dose_size[3] = +50.0f;
        dose_size[4] = -50.0f;
        dose_size[5] = +50.0f;
        num_multi_sphere = 33;
    }
};

PLMUTIL_API void synthetic_mha (Rt_study *rtds, Synthetic_mha_parms *parms);

#endif
