/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "gdcm_series.h"

void
do_dicom_series_test (char *dicom_dir)
{
    gdcm_series_test (dicom_dir);
}

int
main(int argc, char *argv[])
{
    char *dicom_dir;
    if (argc == 2) {
	dicom_dir = argv[1];
    } else {
	printf ("Usage: dicom_series_test dicom_dir\n");
	exit (1);
    }

    do_dicom_series_test (dicom_dir);
    return 0;
}
