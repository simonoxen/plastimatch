/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "gdcmFile.h"
#include "gdcmGlobal.h"
#include "gdcmSeqEntry.h"
#include "gdcmSQItem.h"
#include "gdcm_rtss.h"
#include "plm_path.h"
#include "cxt_io.h"

class Program_parms {
public:
    char dicom_dir[_MAX_PATH];
    char output_fn[_MAX_PATH];
    char cxt_fn[_MAX_PATH];

public:
    Program_parms () {
	memset (this, 0, sizeof(Program_parms));
    }
};

void
do_cxt_to_dicom_rtss (Program_parms *parms)
{
    Cxt_structure_list structures;

    cxt_initialize (&structures);

    cxt_read (&structures, parms->cxt_fn);
    gdcm_rtss_save (&structures, parms->output_fn, parms->dicom_dir);

    //    cxt_write (&structures, "junk.cxt", true);

    cxt_destroy (&structures);
}

void
print_usage (void)
{
    printf ("Usage: cxt_to_dicom_rtss [options] cxt_file\n"
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
    strncpy (parms->cxt_fn, argv[0], _MAX_PATH);

    if (!parms->output_fn) {
	strncpy (parms->output_fn, "rtss_output.dcm", _MAX_PATH);
    }
}

int
main(int argc, char *argv[])
{
    Program_parms parms;
#if defined (commentout)
    char *cxt_fn, *rtss_fn;
    if (argc == 3) {
	cxt_fn = argv[1];
	rtss_fn = argv[2];
    } else {
	printf ("Usage: cxt_to_dicom_rtss cxt_file dicom_rtss_file\n");
	exit (1);
    }
#endif

    parse_args (&parms, argc, argv);

    do_cxt_to_dicom_rtss (&parms);
    return 0;
}
