/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "compiler_warnings.h"
#include "dcmtk_file.h"
#include "dcmtk_series_set.h"
#include "print_and_exit.h"

Dcmtk_series_set::Dcmtk_series_set ()
{
}

Dcmtk_series_set::Dcmtk_series_set (const char* dicom_dir)
{
    this->insert_directory (dicom_dir);
}

Dcmtk_series_set::~Dcmtk_series_set ()
{
    /* Delete Dicom_series objects in map */
    Dcmtk_series_map::iterator it;
    for (it = m_smap.begin(); it != m_smap.end(); ++it) {
	delete (*it).second;
    }
}

void
Dcmtk_series_set::insert_file (const char* fn)
{
    Dcmtk_file *df = new Dcmtk_file (fn);

    const char *c = NULL;
    c = df->get_cstr (DCM_SeriesInstanceUID);
    if (!c) {
	/* No SeriesInstanceUID? */
	delete df;
	return;
    }

    Dcmtk_series_map::iterator it;
    it = m_smap.find (std::string(c));
    if (it == m_smap.end()) {
	std::pair<Dcmtk_series_map::iterator,bool> ret 
	    = m_smap.insert (Dcmtk_series_map_pair (std::string(c), 
		    new Dcmtk_series()));
	if (ret.second == false) {
	    print_and_exit (
		"Error inserting UID %s into dcmtk_series_map.\n", c);
	}
	it = ret.first;
    }
    Dcmtk_series *ds = (*it).second;
    ds->insert (df);
}

void
Dcmtk_series_set::insert_directory (const char* dir)
{
    OFBool recurse = OFFalse;
    OFList<OFString> input_files;

    OFStandard::searchDirectoryRecursively (
	dir, input_files, "", "", recurse);

    OFListIterator(OFString) if_iter = input_files.begin();
    OFListIterator(OFString) if_last = input_files.end();
    while (if_iter != if_last) {
	const char *current = (*if_iter++).c_str();
	this->insert_file (current);
    }
}

void
Dcmtk_series_set::debug (void) const
{
    Dcmtk_series_map::const_iterator it;
    for (it = m_smap.begin(); it != m_smap.end(); ++it) {
	const std::string& key = (*it).first;
	const Dcmtk_series *ds = (*it).second;
	UNUSED_VARIABLE (ds);
	printf ("SeriesInstanceUID = %s\n", key.c_str());
	ds->debug ();
    }
}

void
dcmtk_series_set_test (char *dicom_dir)
{
    Dcmtk_series_set dss;
    printf ("Searching directory: %s\n", dicom_dir);
    dss.insert_directory (dicom_dir);
    dss.debug ();
}
