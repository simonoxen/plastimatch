/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdlib.h>
#include <stdio.h>

#include "plm_config.h"
#include "plm_path.h"
#include "getopt.h"
#include "cxt_io.h"
#include "xio_io.h"

class Program_parms {
public:
    char xio_dir[_MAX_PATH];
    char dicom_dir[_MAX_PATH];
    char output_cxt_fn[_MAX_PATH];
    float x_adj;
    float y_adj;

public:
    Program_parms () {
	memset (this, 0, sizeof(Program_parms));
    }
};

void
print_usage (void)
{
    printf ("Usage: xio_to_cxt [options] xio-directory\n"
	    "Optional:\n"
	    "    --dicom-dir=directory\n"
	    "    --output=filename\n"
	    "    -x-adj=float\n"
	    "    -y-adj=float\n");
}

void
parse_args (Program_parms* parms, int argc, char* argv[])
{
    int ch, rc;

    static struct option longopts[] = {
	{ "dicom-dir",      required_argument,      NULL,           1 },
	{ "dicom_dir",      required_argument,      NULL,           1 },
	{ "output",	    required_argument,      NULL,           2 },
	{ "x-adj",	    required_argument,      NULL,           3 },
	{ "x_adj",	    required_argument,      NULL,           3 },
	{ "y-adj",	    required_argument,      NULL,           4 },
	{ "y_adj",	    required_argument,      NULL,           4 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    strncpy (parms->dicom_dir, optarg, _MAX_PATH);
	    break;
	case 2:
	    strncpy (parms->output_cxt_fn, optarg, _MAX_PATH);
	    break;
	case 3:
	    rc = sscanf (optarg, "%f", &parms->x_adj);
	    if (rc != 1) {
		fprintf (stderr, "Error.  --x-adj requires a floating point argument.");
		exit (1);
	    }
	    break;
	case 4:
	    rc = sscanf (optarg, "%f", &parms->y_adj);
	    if (rc != 1) {
		fprintf (stderr, "Error.  --y-adj requires a floating point argument.");
		exit (1);
	    }
	    break;
	default:
	    break;
	}
    }

    argc -= optind;
    argv += optind;
    if (argc != 1) {
	print_usage ();
    }
    strncpy (parms->xio_dir, argv[0], _MAX_PATH);

    if (!parms->output_cxt_fn[0]) {
	strncpy (parms->output_cxt_fn, "output.cxt", _MAX_PATH);
    }
}

void
do_xio_to_cxt (Program_parms *parms)
{
    Cxt_structure_list structures;

    xio_load_structures (&structures, parms->xio_dir, parms->x_adj, 
			 parms->y_adj);

    /* Write out the cxt */
    cxt_write (&structures, parms->output_cxt_fn);
}

int 
main (int argc, char* argv[]) 
{
    Program_parms parms;

    parse_args (&parms, argc, argv);

    do_xio_to_cxt (&parms);

    return 0;
}
