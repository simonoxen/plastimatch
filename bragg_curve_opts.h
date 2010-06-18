/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bragg_curve_opts_h_
#define _bragg_curve_opts_h_

typedef struct bragg_curve_options Bragg_curve_options;
struct bragg_curve_options {
    char* output_file;
};

void parse_args (Bragg_curve_options* options, int argc, char* argv[]);

#endif
