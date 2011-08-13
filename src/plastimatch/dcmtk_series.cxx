/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#define HAVE_CONFIG_H 1               /* Needed for debian */
#include "dcmtk/config/osconfig.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_file.h"
#include "dcmtk_series.h"
#include "print_and_exit.h"

void
dcmtk_series_test (char *dicom_dir)
{
    OFBool recurse = OFFalse;
    OFList<OFString> input_files;

    printf ("Searching directory: %s\n", dicom_dir);

    OFStandard::searchDirectoryRecursively (
	dicom_dir, input_files, "", "", recurse);

    if (input_files.empty()) {
	print_and_exit ("Sorry.  Directory %s is empty.\n", dicom_dir);
    }

    OFListIterator(OFString) if_iter = input_files.begin();
    OFListIterator(OFString) if_last = input_files.end();
    while (if_iter != if_last) {
	const char *current = (*if_iter++).c_str();
	printf ("File: %s\n", current);

	Dcmtk_file df;
	df.load (current);
    }
}
