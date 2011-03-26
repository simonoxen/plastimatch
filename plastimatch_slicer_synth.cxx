/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "plastimatch_slicer_synthCLP.h"
#include "itk_image.h"
#include "itk_image_save.h"
#include "synthetic_mha.h"

int 
main (int argc, char * argv [])
{
    PARSE_ARGS;

    Synthetic_mha_parms sm_parms;
    if (plmslc_dim.size() >= 3) {
	sm_parms.dim[0] = plmslc_dim[0];
	sm_parms.dim[1] = plmslc_dim[1];
	sm_parms.dim[2] = plmslc_dim[2];
    } else if (plmslc_dim.size() >= 1) {
	sm_parms.dim[0] = plmslc_dim[0];
	sm_parms.dim[1] = plmslc_dim[0];
	sm_parms.dim[2] = plmslc_dim[0];
    }
    if (plmslc_origin.size() >= 3) {
	sm_parms.origin[0] = plmslc_origin[0];
	sm_parms.origin[1] = plmslc_origin[1];
	sm_parms.origin[2] = plmslc_origin[2];
    } else if (plmslc_origin.size() >= 1) {
	sm_parms.origin[0] = plmslc_origin[0];
	sm_parms.origin[1] = plmslc_origin[0];
	sm_parms.origin[2] = plmslc_origin[0];
    }
    if (plmslc_spacing.size() >= 3) {
	sm_parms.spacing[0] = plmslc_spacing[0];
	sm_parms.spacing[1] = plmslc_spacing[1];
	sm_parms.spacing[2] = plmslc_spacing[2];
    } else if (plmslc_spacing.size() >= 1) {
	sm_parms.spacing[0] = plmslc_spacing[0];
	sm_parms.spacing[1] = plmslc_spacing[0];
	sm_parms.spacing[2] = plmslc_spacing[0];
    }
    if (plmslc_pattern == "Gauss") {
	sm_parms.pattern = PATTERN_GAUSS;
    } else if (plmslc_pattern == "Rectangle") {
	sm_parms.pattern = PATTERN_RECT;
    } else if (plmslc_pattern == "Sphere") {
	sm_parms.pattern = PATTERN_SPHERE;
    }

    /* Create images */
    FloatImageType::Pointer img;
    if (plmslc_output_one != "" && plmslc_output_one != "None") {
	img = synthetic_mha (&sm_parms);
	itk_image_save_float (img, plmslc_output_one.c_str());
    }
    if (plmslc_output_two != "" && plmslc_output_two != "None") {
	img = synthetic_mha (&sm_parms);
	itk_image_save_float (img, plmslc_output_two.c_str());
    }

    return EXIT_SUCCESS;
}
