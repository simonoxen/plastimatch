/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _check_grad_opts_h_
#define _check_grad_opts_h_

#include "plm_config.h"
#include "bspline.h"

#define CHECK_GRAD_PROCESS_FWD        1
#define CHECK_GRAD_PROCESS_BKD        2
#define CHECK_GRAD_PROCESS_CTR        3
#define CHECK_GRAD_PROCESS_LINE       4


class Check_grad_opts {
public:
    char* fixed_fn;
    char* moving_fn;
    char* input_xf_fn;
    char* output_fn;
    float factr;
    float pgtol;
    float step_size;
    int line_range[2];
    int vox_per_rgn[3];
    int process;
    int random;
    float random_range[2];
    Bspline_parms parms;
public:
    Check_grad_opts () {
	fixed_fn = 0;
	moving_fn = 0;
	input_xf_fn = 0;
	output_fn = 0;
	factr = 0;
	pgtol = 0;
	step_size = 1e-4;
	line_range[0] = 0;
	line_range[1] = 30;
	for (int d = 0; d < 3; d++) {
	    vox_per_rgn[d] = 15;
	}
	process = CHECK_GRAD_PROCESS_FWD;
	random = 0;
	random_range[0] = 0;
	random_range[1] = 0;
    }
};

#if defined __cplusplus
extern "C" {
#endif

void
check_grad_opts_parse_args (Check_grad_opts* options, 
    int argc, char* argv[]);

#if defined __cplusplus
}
#endif

#endif
