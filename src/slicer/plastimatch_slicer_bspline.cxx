/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <vector>
#include "file_util.h"
#include "plastimatch_slicer_bsplineCLP.h"
#include "plm_stages.h"

int 
main (int argc, char * argv [])
{
    PARSE_ARGS;

    std::ostringstream command_string;

    command_string <<
	"[GLOBAL]\n"
	"fixed=" << plmslc_fixed_volume << "\n"
	"moving=" << plmslc_moving_volume << "\n";

    if (plmslc_output_warped != "" && plmslc_output_warped != "None") {
	command_string << 
	    "img_out=" << plmslc_output_warped << "\n";
    }

    /* GCS FIX: You can specify two output bspline xforms, but not 
       (yet) two output vector field files */
    if (plmslc_output_vf != "" && plmslc_output_vf != "None") {
	command_string << 
	    "vf_out=" << plmslc_output_vf << "\n";
    }
    else if (plmslc_output_vf_f != "" && plmslc_output_vf_f != "None") {
	command_string << 
	    "vf_out=" << plmslc_output_vf_f << "\n";
    }

    if (plmslc_output_bsp != "" && plmslc_output_bsp != "None") {
	command_string << 
	    "xf_out_itk=true\n"
	    "xf_out=" << plmslc_output_bsp << "\n";
    }
    if (plmslc_output_bsp_f != "" && plmslc_output_bsp_f != "None") {
	command_string << 
	    "xf_out_itk=true\n"
	    "xf_out=" << plmslc_output_bsp_f << "\n";
    }

    /* Stage 0 */
    if (enable_stage_0) {
	command_string << 
	    "[STAGE]\n"
	    "metric=" 
	    << metric << "\n"
	    "xform=translation\n"
	    "optim=rsg\n"
	    "impl=itk\n"
	    "max_its=" << stage_0_its << "\n"
	    "convergence_tol=5\n"
	    "grad_tol=1.5\n"
	    "res=" 
	    << stage_0_resolution[0] << " "
	    << stage_0_resolution[1] << " "
	    << stage_0_resolution[2] << "\n";
    }

    /* Stage 1 */
    command_string << 
	"[STAGE]\n"
	"metric=" 
	<< metric << "\n"
	"xform=bspline\n"	
	"optim=lbfgsb\n"
	"impl=plastimatch\n"
	"threading=" 
	<< (strcmp (hardware.c_str(),"CPU") ? "cuda" : "openmp")
	<< "\n"
	"max_its=" << stage_1_its << "\n"
	"convergence_tol=5\n"
	"grad_tol=1.5\n"
	"res=" 
	<< stage_1_resolution[0] << " "
	<< stage_1_resolution[1] << " "
	<< stage_1_resolution[2] << "\n"
	"grid_spac="
	<< stage_1_grid_size << " "
	<< stage_1_grid_size << " "
	<< stage_1_grid_size << "\n"
	;
    
    if (enable_stage_2) {
	command_string << 
	    "[STAGE]\n"
	    "max_its=" << stage_2_its << "\n"
	    "res=" 
	    << stage_2_resolution[0] << " "
	    << stage_2_resolution[1] << " "
	    << stage_2_resolution[2] << "\n"
	    "grid_spac="
	    << stage_2_grid_size << " "
	    << stage_2_grid_size << " "
	    << stage_2_grid_size << "\n"
	    ;
    }

    std::cout << command_string.str() << "\n";

    /* Go back to beginning of file */
    Registration_Parms regp;
    if (regp.set_command_string (command_string.str()) < 0) {
	return EXIT_FAILURE;
    }

    do_registration (&regp);

    return EXIT_SUCCESS;
}
