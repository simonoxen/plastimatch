/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "reg_opts.h"

void
print_usage (void)
{
    printf (
    "Usage: reg [options] [input]\n"
    "Options:\n"
    " -f implementation        Choose implementation (single letter: a, b, c, etc)\n"
    " -s \"i j k\"               Control point spacing for -V (default: 15 15 15)\n"
    "\n"
    "Input (choose one):\n"
    " -V input_vector_field    Specify input vector field mha\n"
    " -C input_coefficients    Specify input B-spline coefficients\n\n"
    );
    exit(1);
}

void
reg_opts_parse_args (Reg_options* options, int argc, char* argv[])
{
    int i,rc;
    Reg_parms* parms = &options->parms;

    for (i=1; i < argc; i++) {
    if (argv[i][0] != '-') break;
    /* BEHAVIOR OPTIONS */
    if (!strcmp (argv[i], "-f")) {
        if (i == (argc-1) || argv[i+1][0] == '-') {
        fprintf(stderr, "option %s requires an argument\n", argv[i]);
        exit(1);
        }
        i++;
        parms->implementation = argv[i][0];
    }
    else if (!strcmp (argv[i], "-s")) {
        if (i == (argc-1) || argv[i+1][0] == '-') {
        fprintf(stderr, "option %s requires an argument\n", argv[i]);
        exit(1);
        }
        i++;
        rc = sscanf (argv[i], "%d %d %d", 
        &options->vox_per_rgn[0],
        &options->vox_per_rgn[1],
        &options->vox_per_rgn[2]);
        if (rc == 1) {
        options->vox_per_rgn[1] = options->vox_per_rgn[0];
        options->vox_per_rgn[2] = options->vox_per_rgn[0];
        } else if (rc != 3) {
        print_usage ();
        }
    }
    /* INPUT OPTIONS */
    else if (!strcmp (argv[i], "-V")) {
        if (i == (argc-1) || argv[i+1][0] == '-') {
        fprintf(stderr, "option %s expects path to mha file\n", argv[i]);
        exit(1);
        }
        i++;
        options->input_vf_fn = strdup (argv[i]);
        if (options->input_xf_fn != 0) {
            printf ("Error: Cannot specify both vector field AND coefficient input\n");
            exit (1);
        } else {
            printf ("Using Vector Field: %s\n", options->input_vf_fn);
        }
    }
    else if (!strcmp (argv[i], "-C")) {
        if (i == (argc-1) || argv[i+1][0] == '-') {
        fprintf(stderr, "option %s expects path to xform file\n", argv[i]);
        exit(1);
        }
        i++;
        options->input_xf_fn = strdup (argv[i]);
        if (options->input_vf_fn != 0) {
            printf ("Error: Cannot specify both vector field AND coefficient input\n");
            exit (1);
        } else {
            printf ("Using Coefficients: %s\n", options->input_xf_fn);
        }
    }
    else {
        print_usage ();
        break;
    }
    } /* for (i < argc) */

    if (options->input_xf_fn == 0 && options->input_vf_fn == 0) {
        print_usage ();
    }
}
