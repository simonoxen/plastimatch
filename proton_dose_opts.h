/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_dose_opts_h_
#define _proton_dose_opts_h_

#include "threading.h"

typedef struct proton_dose_options Proton_dose_options;
struct proton_dose_options {
    enum Threading threading;

    float src[3];                    /* Beam source */
    float isocenter[3];              /* Beam target */
    float vup[3];                    /* Aperture orientation */

    float scale;
    float ray_step;                  /* Uniform ray step size (mm) */
    char* input_pep_fn;              /* Proton energy profile */
    char* input_fn;
    char* output_fn;
    int debug;
};

void proton_dose_parse_args (Proton_dose_options* options, 
    int argc, char* argv[]);

#endif
