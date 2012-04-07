/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "astroid_dose.h"
#include "bstring_util.h"
#include "cxt_extract.h"
#include "file_util.h"
#if GDCM_VERSION_1
#include "gdcm1_dose.h"
#include "gdcm1_series.h"
#include "gdcm1_rtss.h"
#endif
#include "mc_dose.h"
#include "referenced_dicom_dir.h"
#include "rtds.h"
#include "rtss.h"
#include "ss_list_io.h"
#include "xio_ct.h"
#include "xio_demographic.h"
#include "xio_dir.h"
#include "xio_dose.h"
#include "xio_studyset.h"
#include "xio_structures.h"

Rtds::Rtds ()
{
    m_img = 0;
    m_rtss = 0;
    m_dose = 0;
#if GDCM_VERSION_1
    m_gdcm_series = 0;
#endif
    m_meta.create_anonymous ();

    m_xio_transform = (Xio_ct_transform*) malloc (sizeof (Xio_ct_transform));
    xio_ct_get_transform(&m_meta, m_xio_transform);

    strcpy (m_xio_dose_input, "\0");
}

Rtds::~Rtds ()
{
    if (m_img) {
	delete m_img;
    }
    if (m_rtss) {
	delete m_rtss;
    }
    if (m_dose) {
	delete m_dose;
    }
#if GDCM_VERSION_1
    if (m_gdcm_series) {
	delete m_gdcm_series;
    }
#endif
    if (m_xio_transform) {
	free (m_xio_transform);
    }
}

void
Rtds::load_dicom (const char *dicom_dir)
{
    if (!dicom_dir) {
	return;
    }

#if PLM_DCM_USE_DCMTK
    this->load_dcmtk (dicom_dir);
#else
    this->load_gdcm (dicom_dir);
#endif
}

void
Rtds::load_dicom_dir (const char *dicom_dir)
{
    const char *dicom_dir_tmp;  /* In case dicom_dir is a file, not dir */

    if (is_directory (dicom_dir)) {
	dicom_dir_tmp = dicom_dir;
    } else {
	dicom_dir_tmp = file_util_dirname (dicom_dir);
    }

    this->load_dicom (dicom_dir_tmp);

    if (dicom_dir_tmp != dicom_dir) {
	free ((void*) dicom_dir_tmp);
    }
}

void
Rtds::load_xio (
    const char *xio_dir,
    Referenced_dicom_dir *rdd
)
{
    Xio_dir xd (xio_dir);
    Xio_patient *xpd;
    Xio_studyset_dir *xsd;
    Xio_plan_dir *xtpd;

    if (xd.num_patients() <= 0) {
	print_and_exit ("Error, xio num_patient_dir = %d\n", 
	    xd.num_patients());
    }
    xpd = xd.patient_dir[0];
    if (xd.num_patients() > 1) {
	printf ("Warning: multiple patients found in xio directory.\n"
	    "Defaulting to first directory: %s\n", 
	    (const char*) xpd->m_path);
    }

    if (xpd->num_plan_dir > 0) {

	/* When plans exist, load the first plan */
	xtpd = &xpd->plan_dir[0];
	if (xpd->num_studyset_dir > 1) {
	    printf ("Warning: multiple plans found in xio patient directory.\n"
		"Defaulting to first directory: %s\n", xtpd->path);
	}

	/* Load the summed XiO dose file */
	this->m_dose = new Plm_image ();
	printf ("calling xio_dose_load\n");
	std::string xio_dose_file = std::string(xtpd->path) + "/dose.1";
	strncpy(this->m_xio_dose_input, xio_dose_file.c_str(), _MAX_PATH);
	xio_dose_load (this->m_dose, &m_meta, xio_dose_file.c_str());

	/* Find studyset associated with plan */
	xsd = xio_plan_dir_get_studyset_dir (xtpd);

    } else {

	/* No plans exist, load only studyset */

	if (xpd->num_studyset_dir <= 0) {
	    print_and_exit ("Error, xio patient has no studyset.");
	}

	printf ("Warning: no plans found, only loading studyset.");

	xsd = &xpd->studyset_dir[0];
	if (xpd->num_studyset_dir > 1) {
	    printf (
		"Warning: multiple studyset found in xio patient directory.\n"
		"Defaulting to first directory: %s\n", xsd->path);
	}
    }

    printf("path is :: %s\n", xsd->path);

    /* Load the XiO studyset slice list */
    Xio_studyset xst (xsd->path);

    /* Load the XiO studyset CT images */
    this->m_img = new Plm_image;
    xio_ct_load (this->m_img, &xst);

    /* Load the XiO studyset structure set */
    this->m_rtss = new Rtss (this);
    this->m_rtss->load_xio (xst);

    /* Apply XiO CT geometry to structures */
    if (this->m_rtss->m_cxt) {
	printf ("calling cxt_set_geometry_from_plm_image\n");
	this->m_rtss->m_cxt->set_geometry_from_plm_image (this->m_img);
    }

    /* Load demographics */
    if (xpd->m_demographic_fn.not_empty()) {
	Xio_demographic demographic ((const char*) xpd->m_demographic_fn);
	if (demographic.m_patient_name.not_empty()) {
	    this->m_meta.set_metadata (0x0010, 0x0010, 
		(const char*) demographic.m_patient_name);
	}
	if (demographic.m_patient_id.not_empty()) {
	    this->m_meta.set_metadata (0x0010, 0x0020, 
		(const char*) demographic.m_patient_id);
	}
    }

    /* If referenced DICOM CT is provided,  the coordinates will be
       transformed from XiO to DICOM LPS  with the same origin as the
       original CT.

       Otherwise, the XiO CT will be saved as DICOM and the structures
       will be associated to those slices. The coordinates will be
       transformed to DICOM LPS based on the patient position metadata
       and the origin will remain the same. */

    if (this->m_img) {
	if (m_rdd.m_loaded) {
	    /* Determine transformation based original DICOM */
	    xio_ct_get_transform_from_rdd
		(this->m_img, &m_meta, rdd, this->m_xio_transform);
	} else {
    	    /* Determine transformation based on patient position */
	    xio_ct_get_transform (&m_meta, this->m_xio_transform);
	}
    }

    if (this->m_img) {
	xio_ct_apply_transform (this->m_img, this->m_xio_transform);
    }
    if (this->m_rtss->m_cxt) {
	xio_structures_apply_transform (this->m_rtss->m_cxt, 
	    this->m_xio_transform);
    }
    if (this->m_dose) {
	xio_dose_apply_transform (this->m_dose, this->m_xio_transform);
    }
}

void
Rtds::load_ss_img (const char *ss_img, const char *ss_list)
{
    this->m_rtss = new Rtss (this);
    this->m_rtss->load (ss_img, ss_list);
}

void
Rtds::load_dose_img (const char *dose_img)
{
    if (this->m_dose) {
	delete this->m_dose;
    }
    if (dose_img) {
	this->m_dose = plm_image_load_native (dose_img);
    }
}

void
Rtds::load_rdd (const char *rdd)
{
    m_rdd.load (rdd);

    if (m_rdd.m_loaded) {
	/* Default to patient position in referenced DICOM */
	m_meta.set_metadata(0x0018, 0x5100,
	    m_rdd.m_demographics.get_metadata(0x0018, 0x5100));
	xio_ct_get_transform(&m_meta, m_xio_transform);

	/* Default to patient name/ID/sex in referenced DICOM */
	m_meta.set_metadata(0x0010, 0x0010,
	    m_rdd.m_demographics.get_metadata(0x0010, 0x0010));
	m_meta.set_metadata(0x0010, 0x0020,
	    m_rdd.m_demographics.get_metadata(0x0010, 0x0020));
	m_meta.set_metadata(0x0010, 0x0040,
	    m_rdd.m_demographics.get_metadata(0x0010, 0x0040));
    }
}

void
Rtds::load_dose_xio (const char *dose_xio)
{
    if (this->m_dose) {
	delete this->m_dose;
    }
    if (dose_xio) {
	strncpy(this->m_xio_dose_input, dose_xio, _MAX_PATH);
	this->m_dose = new Plm_image ();
	xio_dose_load (this->m_dose, &m_meta, dose_xio);
	xio_dose_apply_transform (this->m_dose, this->m_xio_transform);
    }
}

void
Rtds::load_dose_astroid (const char *dose_astroid)
{
    if (this->m_dose) {
	delete this->m_dose;
    }
    if (dose_astroid) {
	this->m_dose = new Plm_image ();
	astroid_dose_load (this->m_dose, &m_meta, dose_astroid);
	astroid_dose_apply_transform (this->m_dose, this->m_xio_transform);
    }
}

void
Rtds::load_dose_mc (const char *dose_mc)
{
    if (this->m_dose) {
	delete this->m_dose;
    }
    if (dose_mc) {
	this->m_dose = new Plm_image ();
	mc_dose_load (this->m_dose, dose_mc);
	mc_dose_apply_transform (this->m_dose, this->m_xio_transform);
    }
}

void
Rtds::save_dicom (const char *dicom_dir)
{
    if (!dicom_dir) {
	return;
    }

#if PLM_DCM_USE_DCMTK
    this->save_dcmtk (dicom_dir);
#else
    this->save_gdcm (dicom_dir);
#endif
}

void 
Rtds::set_user_metadata (std::vector<std::string>& metadata)
{
    std::vector<std::string>::iterator it = metadata.begin();
    while (it < metadata.end()) {
	const std::string& str = (*it);
	size_t eq_pos = str.find_first_of ('=');
	if (eq_pos != std::string::npos) {
	    std::string key = str.substr (0, eq_pos);
	    std::string val = str.substr (eq_pos+1);
	    m_meta.set_metadata (key, val);
	}
	++it;
    }

    xio_ct_get_transform(&(m_meta), m_xio_transform);
}

void 
Rtds::set_dose (Plm_image *pli)
{
    if (m_dose) delete m_dose;
    m_dose = pli;
}
