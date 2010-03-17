/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dvh_main_h_
#define _dvh_main_h_

#include "plm_config.h"
#include "plm_path.h"

class Dvh_parms {
public:
    char input_ss_img[_MAX_PATH];
    char input_ss_list[_MAX_PATH];
    char input_dose[_MAX_PATH];
    char output_csv[_MAX_PATH];
public:
    Dvh_parms () {
	input_ss_img[0] = 0;
	input_ss_list[0] = 0;
	input_dose[0] = 0;
	output_csv[0] = 0;
    }
};

void do_command_dvh (int argc, char *argv[]);

#endif
