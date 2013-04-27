/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk/config/osconfig.h"
#include "dcmtk/oflog/oflog.h"

#include "compiler_warnings.h"
#include "dcmtk_file.h"
#include "dcmtk_loader.h"
#include "dcmtk_loader_p.h"
#include "dcmtk_series.h"
#include "dicom_rt_study.h"
#include "file_util.h"
#include "plm_image.h"
#include "print_and_exit.h"
#include "rtss.h"

Dcmtk_loader::Dcmtk_loader ()
{
    d_ptr = new Dcmtk_loader_private;
    init ();
}

Dcmtk_loader::Dcmtk_loader (const char* dicom_path)
{
    d_ptr = new Dcmtk_loader_private;
    init ();

    /* GCS FIX: Need a way to turn this on via configuration. 
       But for now, just unilaterally disable logging. 
       http://support.dcmtk.org/wiki/dcmtk/howto/logprogram */
    OFLog::configure (OFLogger::FATAL_LOG_LEVEL);

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
    d_ptr->ds_rtdose = 0;
    d_ptr->ds_rtss = 0;
}

void
Dcmtk_loader::set_dicom_metadata (Dicom_rt_study::Pointer drs)
{
    d_ptr->m_drs = drs;
}

void
Dcmtk_loader::insert_file (const char* fn)
{
    Dcmtk_file *df = new Dcmtk_file (fn);

    /* Discard non-dicom files */
    if (!df->is_valid()) {
        delete df;
        return;
    }

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
    return &d_ptr->img->m_meta;
}

Volume *
Dcmtk_loader::get_volume ()
{
    if (!d_ptr->img) {
        this->parse_directory ();
    }
    if (!d_ptr->img) {
        return 0;
    }
    return d_ptr->img->get_volume();
}

Plm_image::Pointer
Dcmtk_loader::get_image ()
{
    return d_ptr->img;
}

Rtss::Pointer
Dcmtk_loader::get_rtss ()
{
    return d_ptr->cxt;
}

Plm_image::Pointer
Dcmtk_loader::get_dose ()
{
    return d_ptr->dose;
}

void
Dcmtk_loader::parse_directory (void)
{
    Dcmtk_series_map::iterator it;
    d_ptr->ds_image = 0;
    d_ptr->ds_rtss = 0;
    d_ptr->ds_rtdose = 0;

    /* Loop through all series in directory, and find image, ss, dose */
    size_t best_image_slices = 0;
    for (it = d_ptr->m_smap.begin(); it != d_ptr->m_smap.end(); ++it) {
	const std::string& key = (*it).first;
	Dcmtk_series *ds = (*it).second;
	UNUSED_VARIABLE (key);

	/* Check for rtstruct */
	if (!d_ptr->ds_rtss && ds->get_modality() == "RTSTRUCT") {
	    printf ("Found RTSTUCT, UID=%s\n", key.c_str());
	    d_ptr->ds_rtss = ds;
	    continue;
	}

	/* Check for rtdose */
	if (!d_ptr->ds_rtdose && ds->get_modality() == "RTDOSE") {
	    printf ("Found RTDOSE, UID=%s\n", key.c_str());
	    d_ptr->ds_rtdose = ds;
	    continue;
	}

	/* Check for image.  An image is anything with a PixelData.
           Current heuristic: load the image with the most slices
           (as determined by the number of files) */
	bool rc = ds->get_uint16_array (DCM_PixelData, 0, 0);
        if (rc) {
            size_t num_slices = ds->get_number_of_files ();
            if (num_slices > best_image_slices) {
                best_image_slices = num_slices;
                d_ptr->ds_image = ds;
            }
	    continue;
	}
    }

    /* GCS FIX: need additional logic that checks if ss & dose 
       refer to the image.  The below logic doesn't do anything. */
    std::string referenced_uid = "";
    if (d_ptr->ds_rtss) {
	referenced_uid = d_ptr->ds_rtss->get_referenced_uid ();
    }

    /* Load image */
    if (d_ptr->ds_image) {
        d_ptr->ds_image->set_dicom_metadata (d_ptr->m_drs);
        d_ptr->img = d_ptr->ds_image->load_plm_image ();
    }

    /* Load rtss */
    if (d_ptr->ds_rtss) {
        this->rtss_load ();
    }

    /* Load dose */
    if (d_ptr->ds_rtdose) {
        this->rtdose_load ();
    }
}

ShortImageType::Pointer 
dcmtk_load (const char *dicom_dir)
{
    ShortImageType::Pointer img = ShortImageType::New ();
    
    return img;
}

void
Dcmtk_loader::set_dose (Plm_image::Pointer dose)
{
    d_ptr->dose = dose;
}
