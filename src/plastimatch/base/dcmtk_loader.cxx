/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "compiler_warnings.h"
#include "dcmtk_file.h"
#include "dcmtk_loader.h"
#include "dcmtk_loader_p.h"
#include "dcmtk_series.h"
#include "dicom_rt_study.h"
#include "file_util.h"
#include "plm_image.h"
#include "print_and_exit.h"

Dcmtk_loader::Dcmtk_loader ()
{
    d_ptr = new Dcmtk_loader_private;
    init ();
}

Dcmtk_loader::Dcmtk_loader (const char* dicom_path)
{
    d_ptr = new Dcmtk_loader_private;
    init ();
    if (is_directory (dicom_path)) {
        this->insert_directory (dicom_path);
    } else {
        this->insert_file (dicom_path);
    }
}

Dcmtk_loader::~Dcmtk_loader ()
{
    delete d_ptr;
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
Dcmtk_loader::set_rt_study (Dicom_rt_study *drs)
{
    d_ptr->m_drs = drs;
}

void
Dcmtk_loader::insert_file (const char* fn)
{
    Dcmtk_file *df = new Dcmtk_file (fn);

    /* Get the SeriesInstanceUID */
    const char *c = NULL;
    c = df->get_cstr (DCM_SeriesInstanceUID);
    if (!c) {
	/* No SeriesInstanceUID? */
	delete df;
	return;
    }

    /* Look for the SeriesInstanceUID in the map */
    Dcmtk_series_map::iterator it;
    it = d_ptr->m_smap.find (std::string(c));

    /* If we didn't find the UID, add a new entry into the map */
    if (it == d_ptr->m_smap.end()) {
	std::pair<Dcmtk_series_map::iterator,bool> ret 
	    = d_ptr->m_smap.insert (Dcmtk_series_map_pair (std::string(c), 
		    new Dcmtk_series()));
	if (ret.second == false) {
	    print_and_exit (
		"Error inserting UID %s into dcmtk_series_map.\n", c);
	}
	it = ret.first;
    }

    /* Add the file to the Dcmtk_series object for this UID */
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
    for (it = d_ptr->m_smap.begin(); it != d_ptr->m_smap.end(); ++it) {
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
    for (it = d_ptr->m_smap.begin(); it != d_ptr->m_smap.end(); ++it) {
	const std::string& key = (*it).first;
	const Dcmtk_series *ds = (*it).second;
	UNUSED_VARIABLE (ds);
	printf ("SeriesInstanceUID = %s\n", key.c_str());
	ds->debug ();
    }
}

Metadata *
Dcmtk_loader::get_metadata ()
{
    return &this->img->m_meta;
}

Volume *
Dcmtk_loader::get_volume ()
{
    if (!this->img) {
        this->parse_directory ();
    }
    if (!this->img) {
        return 0;
    }
    return this->img->get_volume();
}

Plm_image *
Dcmtk_loader::steal_plm_image ()
{
    /* Transfer ownership to caller */
    Plm_image *tmp = this->img;
    this->img = 0;
    return tmp;
}

Rtss_structure_set *
Dcmtk_loader::steal_rtss_structure_set ()
{
    /* Transfer ownership to caller */
    Rtss_structure_set *tmp = this->cxt;
    this->cxt = 0;
    return tmp;
}

Plm_image *
Dcmtk_loader::steal_dose_image ()
{
    /* Transfer ownership to caller */
    Plm_image *tmp = this->dose;
    this->dose = 0;
    return tmp;
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
    for (it = d_ptr->m_smap.begin(); it != d_ptr->m_smap.end(); ++it) {
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
    for (it = d_ptr->m_smap.begin(); it != d_ptr->m_smap.end(); ++it) {
	const std::string& key = (*it).first;
	Dcmtk_series *ds = (*it).second;
	UNUSED_VARIABLE (key);

	/* Skip stuff we're not interested in */
	const std::string& modality = ds->get_modality();
	if (modality == "RTSTRUCT" || modality == "RTDOSE") {
	    continue;
	}

	if (modality == "CT") {
	    printf ("LOADING CT\n");

            /* Load image */
            ds->set_rt_study (d_ptr->m_drs);
	    this->img = ds->load_plm_image ();
	    continue;
	}
    }

    /* Load rtss */
    if (ds_rtss) {
        this->rtss_load ();
    }

    /* Load dose */
    if (ds_rtdose) {
        this->rtdose_load ();
    }
}

ShortImageType::Pointer 
dcmtk_load (const char *dicom_dir)
{
    ShortImageType::Pointer img = ShortImageType::New ();
    
    return img;
}
