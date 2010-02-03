/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "getopt.h"
#include "plm_path.h"
#include "cxt_apply_dicom.h"
#include "cxt_io.h"

class Program_parms {
public:
    char dicom_dir[_MAX_PATH];
    char cxt_fn[_MAX_PATH];

public:
    Program_parms () {
	memset (this, 0, sizeof(Program_parms));
    }
};

void
do_cxt_apply_dicom (Program_parms *parms)
{
    Cxt_structure_list structures;

    cxt_init (&structures);
    cxt_read (&structures, parms->cxt_fn);

    cxt_apply_dicom_dir (&structures, parms->dicom_dir);

    cxt_write (&structures, parms->cxt_fn, true);
}

void
print_usage (void)
{
    printf ("Usage: cxt_apply_dicom dicom_dir cxt_file\n"
	    );
    exit (-1);
}

void
parse_args (Program_parms* parms, int argc, char* argv[])
{
    int ch;
    static struct option longopts[] = {
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	default:
	    break;
	}
    }

    argc -= optind;
    argv += optind;
    if (argc != 2) {
	print_usage ();
    }
    strncpy (parms->dicom_dir, argv[0], _MAX_PATH);
    strncpy (parms->cxt_fn, argv[1], _MAX_PATH);
}

int
main(int argc, char *argv[])
{
    Program_parms parms;

    parse_args (&parms, argc, argv);

    do_cxt_apply_dicom (&parms);
    return 0;
}
