/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>

void
do_dicom_rtss_to_cxt (char *rtss_fn, char *dicom_dir)
{

}

int
main(int argc, char *argv[])
{
    char *rtss_fn, *dicom_dir;
    if (argc == 2) {
	rtss_fn = argv[1];
	dicom_dir = 0;
    } else if (argc == 3) {
	rtss_fn = argv[1];
	dicom_dir = argv[2];
    } else {
	printf ("Usage: dicom_rtss_to_cxt dicom_rtss [ dicom_dir ]\n");
	exit (1);
    }

    do_dicom_rtss_to_cxt (rtss_fn, dicom_dir);
    return 0;
}
