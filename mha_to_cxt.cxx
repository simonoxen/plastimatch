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
#include "getopt.h"
#include "cxt_extract.h"
#include "cxt_io.h"
#include "itk_image.h"

class Program_parms {
public:
    char dicom_dir[_MAX_PATH];
    char output_fn[_MAX_PATH];
    char cxt_reference_fn[_MAX_PATH];
    char mha_fn[_MAX_PATH];

public:
    Program_parms () {
	memset (this, 0, sizeof(Program_parms));
    }
};

void
do_mha_to_cxt (Program_parms *parms)
{
    Cxt_structure_list structures;
    ULongImageType::Pointer image;

    cxt_initialize (&structures);

    printf ("Loading input file...\n");
    image = load_ulong (parms->mha_fn, 0);
    printf ("Done.\n");

    cxt_extract (&structures, image);

    cxt_write (&structures, parms->output_fn, true);

    cxt_destroy (&structures);
}

void
print_usage (void)
{
    printf ("Usage: mha_to_cxt [options] mha_file\n"
	    "Optional:\n"
	    "    --cxt-reference=filename\n"
	    "    --output=filename\n"
	    );
    exit (-1);
}

void
parse_args (Program_parms* parms, int argc, char* argv[])
{
    int ch;
    static struct option longopts[] = {
	{ "cxt-reference",  required_argument,      NULL,           1 },
	{ "cxt_reference",  required_argument,      NULL,           1 },
	{ "output",	    required_argument,      NULL,           2 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    strncpy (parms->cxt_reference_fn, optarg, _MAX_PATH);
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
    strncpy (parms->mha_fn, argv[0], _MAX_PATH);

    if (!parms->output_fn) {
	strncpy (parms->output_fn, "output.cxt", _MAX_PATH);
    }
}

int
main (int argc, char *argv[])
{
    Program_parms parms;

    parse_args (&parms, argc, argv);

    do_mha_to_cxt (&parms);
    return 0;
}
