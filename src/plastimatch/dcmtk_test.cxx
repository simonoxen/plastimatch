/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <string.h>
#include "dcmtk_config.h"
#include "dcmtk/dcmdata/dcuid.h"
#include "dcmtk/ofstd/ofstream.h"

void
dcmtk_series_test (const char *dicom_dir)
{
}

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

    dcmtk_series_test (dicom_dir);

    return 0;
}

