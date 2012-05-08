/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "dcmtk_loader.h"

int
main (int argc, char *argv[])
{
    char *dicom_dir;
    if (argc == 2) {
	dicom_dir = argv[1];
    } else {
	printf ("Usage: dcmtk_test dicom_dir\n");
	exit (1);
    }

#if defined (GCS_FIX)
    dcmtk_series_set_test (dicom_dir);
#endif

    return 0;
}

