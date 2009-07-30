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
#include "readcxt.h"

void
do_cxt_to_dicom_rtss (char *cxt_fn, char *rtss_fn)
{
    Cxt_structure_list structures;

    cxt_initialize (&structures);

    cxt_read (&structures, cxt_fn);
    gdcm_rtss_save (&structures, rtss_fn);

    cxt_write (&structures, "foo.cxt");
}

int
main(int argc, char *argv[])
{
    char *cxt_fn, *rtss_fn;
    if (argc == 3) {
	cxt_fn = argv[1];
	rtss_fn = argv[2];
    } else {
	printf ("Usage: cxt_to_dicom_rtss cxt_file dicom_rtss_file\n");
	exit (1);
    }

    do_cxt_to_dicom_rtss (cxt_fn, rtss_fn);
    return 0;
}
