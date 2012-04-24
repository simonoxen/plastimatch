/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>

#include "libplmimage.h"

int
main (int argc, char *argv[])
{
    char *dicom_dir;
    if (argc == 2) {
	dicom_dir = argv[1];
    } else {
	printf ("Usage: gdcm1_test dicom_dir\n");
	exit (1);
    }

#if GDCM_VERSION_1
    gdcm1_series_test (dicom_dir);
#endif
    return 0;
}
