/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "mha_io.h"
#include "reg_opts.h"
#include "reg.h"

int
main (int argc, char* argv[])
{
    Reg_options options;
    Reg_parms *parms = &options.parms;
    Volume *vf = 0;
    float S = 9999.9f;

    reg_opts_parse_args (&options, argc, argv);

    vf = read_mha (options.input_vf_fn);
    if (!vf) { exit (-1); }



    switch (parms->implementation) {
    case 'a':
        S = vf_regularize_numerical (vf);
        break;
    default:
        printf ("Warning: Using implementation 'a'\n");
        S = vf_regularize_numerical (vf);
        break;
    } /* switch(implementation) */

    printf ("S = %f\n", S);
}
