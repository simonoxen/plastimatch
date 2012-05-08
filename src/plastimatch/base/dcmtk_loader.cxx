/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

//#include "plmutil.h"
#include "plmsys.h"

#include "compiler_warnings.h"
#include "dcmtk_file.h"
#include "dcmtk_loader.h"
#include "plm_image.h"


ShortImageType::Pointer 
dcmtk_load (const char *dicom_dir)
{
    ShortImageType::Pointer img = ShortImageType::New ();
    
    return img;
}

Dcmtk_loader::Dcmtk_loader ()
{
    init ();
}

Dcmtk_loader::Dcmtk_loader (const char* dicom_dir)
{
    init ();
    this->insert_directory (dicom_dir);
}

Dcmtk_loader::~Dcmtk_loader ()
{
    /* Delete Dicom_series objects in map */
    Dcmtk_series_map::iterator it;
    for (it = m_smap.begin(); it != m_smap.end(); ++it) {
	delete (*it).second;
    }
}

void
Dcmtk_loader::init ()
{
    this->ds_rtdose = 0;
    this->ds_rtss = 0;
    this->cxt = 0;
    this->cxt_metadata = 0;
    this->img = 0;
    this->dose = 0;
}

void
Dcmtk_loader::insert_file (const char* fn)
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
Dcmtk_loader::insert_directory (const char* dir)
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
Dcmtk_loader::sort_all (void) 
{
    Dcmtk_series_map::iterator it;
    for (it = m_smap.begin(); it != m_smap.end(); ++it) {
	const std::string& key = (*it).first;
	Dcmtk_series *ds = (*it).second;
	UNUSED_VARIABLE (key);
	ds->sort ();
    }
}

void
Dcmtk_loader::debug (void) const
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
Dcmtk_loader::parse_directory (void)
{
    Dcmtk_series_map::iterator it;
    this->ds_rtdose = 0;
    this->ds_rtss = 0;

    /* First pass: loop through series and find ss, dose */
    /* GCS FIX: maybe need additional pass, make sure ss & dose 
       refer to same CT, in case of multiple ss & dose in same 
       directory */
    for (it = m_smap.begin(); it != m_smap.end(); ++it) {
	const std::string& key = (*it).first;
	Dcmtk_series *ds = (*it).second;
	UNUSED_VARIABLE (key);

	/* Check for rtstruct */
	if (!ds_rtss && ds->get_modality() == "RTSTRUCT") {
	    printf ("Found RTSTUCT, UID=%s\n", key.c_str());
	    ds_rtss = ds;
	    continue;
	}

	/* Check for rtdose */
	if (!ds_rtdose && ds->get_modality() == "RTDOSE") {
	    printf ("Found RTDOSE, UID=%s\n", key.c_str());
	    ds_rtdose = ds;
	    continue;
	}
    }

    /* Check if uid matches refereneced uid of rtstruct */
    std::string referenced_uid = "";
    if (ds_rtss) {
	referenced_uid = ds_rtss->get_referenced_uid ();
    }

    /* Second pass: loop through series and find img */
    for (it = m_smap.begin(); it != m_smap.end(); ++it) {
	const std::string& key = (*it).first;
	Dcmtk_series *ds = (*it).second;
	UNUSED_VARIABLE (key);

	/* Skip stuff we're not interested in */
	const std::string& modality = ds->get_modality();
	if (modality == "RTSTRUCT"
	    || modality == "RTDOSE")
	{
	    continue;
	}

	if (ds->get_modality() == "CT") {
	    printf ("LOADING CT\n");
	    this->img = ds->load_plm_image ();
	    continue;
	}
    }

    if (ds_rtss) {
        this->rtss_load ();
    }

#if defined (GCS_FIX)
    if (ds_rtdose) {
        ds_rtdose->rtdose_load (rtds);
    }
#endif
}
