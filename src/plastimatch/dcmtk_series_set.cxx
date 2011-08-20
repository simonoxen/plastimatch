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
#include "dcmtk_series_set.h"
#include "print_and_exit.h"

Dcmtk_series_set::Dcmtk_series_set ()
{
}

Dcmtk_series_set::~Dcmtk_series_set ()
{
}

void
Dcmtk_series_set::insert_file (const char* fn)
{
    Dcmtk_file df;
    df.load (fn);

    const char *c = NULL;
    c = df.get_cstr (DCM_SeriesInstanceUID);
    if (!c) {
	/* No SeriesInstanceUID? */
	return;
    }

    Dcmtk_series_map::iterator smap_it;
    smap_it = m_smap.find (std::string(c));
    if (smap_it == m_smap.end()) {
	printf ("Not yet found\n");
	m_smap.insert (
	    Dcmtk_series_map_pair (std::string(c), Dcmtk_series()));
    } else {
	printf ("Already found\n");
    }
}

void
dcmtk_series_set_test (char *dicom_dir)
{
    OFBool recurse = OFFalse;
    OFList<OFString> input_files;

    Dcmtk_series_set dss;

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
	dss.insert_file (current);
    }
}
