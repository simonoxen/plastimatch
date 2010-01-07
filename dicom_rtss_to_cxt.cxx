/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "cxt_io.h"
#include "cxt_to_mha.h"
#include "gdcm_rtss.h"
#include "getopt.h"
#include "plm_path.h"

class Program_parms {
public:
    char dicom_dir[_MAX_PATH];
    char output_fn[_MAX_PATH];
    char rtss_fn[_MAX_PATH];

public:
    Program_parms () {
	memset (this, 0, sizeof(Program_parms));
    }
};

void
do_dicom_rtss_to_cxt (Program_parms *parms)
{
    Cxt_structure_list structures;

    cxt_initialize (&structures);
    gdcm_rtss_load (&structures, parms->rtss_fn, parms->dicom_dir);

    cxt_write (&structures, parms->output_fn, true);
}

void
print_usage (void)
{
    printf ("Usage: dicom_rtss_to_cxt [options] rtss_file\n"
	    "Optional:\n"
	    "    --dicom-dir=directory\n"
	    "    --output=filename\n"
	    );
    exit (-1);
}

void
parse_args (Program_parms* parms, int argc, char* argv[])
{
    int ch;
    static struct option longopts[] = {
	{ "dicom-dir",      required_argument,      NULL,           1 },
	{ "dicom_dir",      required_argument,      NULL,           1 },
	{ "output",	    required_argument,      NULL,           2 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    strncpy (parms->dicom_dir, optarg, _MAX_PATH);
	    break;
	case 2:
	    strncpy (parms->output_fn, optarg, _MAX_PATH);
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
    strncpy (parms->rtss_fn, argv[0], _MAX_PATH);

    if (!parms->output_fn[0]) {
	strncpy (parms->output_fn, "output.cxt", _MAX_PATH);
    }
}

int
main(int argc, char *argv[])
{
    Program_parms parms;

    parse_args (&parms, argc, argv);

    do_dicom_rtss_to_cxt (&parms);
    return 0;
}
