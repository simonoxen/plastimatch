/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if GDCM_VERSION_1
#include "gdcm1_dose.h"
#include "gdcm1_series.h"
#endif
#include "astroid_dose.h"
#include "file_util.h"
#include "mc_dose.h"
#include "path_util.h"
#include "plm_image.h"
#include "print_and_exit.h"
#include "rtds.h"
#include "rtds_p.h"
#include "rtss.h"
#include "rtss_structure_set.h"
#include "slice_index.h"
#include "volume.h"
#include "xio_ct.h"
#include "xio_ct_transform.h"
#include "xio_demographic.h"
#include "xio_dir.h"
#include "xio_dose.h"
#include "xio_patient.h"
#include "xio_structures.h"

Rtds::Rtds ()
{
    d_ptr = new Rtds_private;
    m_img = 0;
    m_rtss = 0;
    m_dose = 0;
    m_rdd = new Slice_index;
}

Rtds::~Rtds ()
{
    delete d_ptr;
    if (m_img) {
        delete m_img;
    }
    if (m_rtss) {
        delete m_rtss;
    }
    if (m_dose) {
        delete m_dose;
    }
    delete m_rdd;
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
Rtds::load_dicom_rtss (const char *dicom_path)
{
    if (this->m_rtss) {
        delete this->m_rtss;
    }
#if PLM_DCM_USE_DCMTK
    this->load_dcmtk (dicom_path);
#elif GDCM_VERSION_1
    this->m_rtss = new Rtss (this);
    this->m_rtss->load_gdcm_rtss (dicom_path, this->m_rdd);
#else
    /* Do nothing */
#endif
}

void
Rtds::load_dicom_dose (const char *dicom_path)
{
#if PLM_DCM_USE_DCMTK
    this->load_dcmtk (dicom_path);
#elif GDCM_VERSION_1
    this->m_dose = gdcm1_dose_load (0, dicom_path);
#else
    /* Do nothing */
#endif
}

void
Rtds::load_xio (
    const char *xio_dir,
    Slice_index *rdd
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
        d_ptr->m_xio_dose_filename = std::string(xtpd->path) + "/dose.1";
        xio_dose_load (this->m_dose, d_ptr->m_meta,
            d_ptr->m_xio_dose_filename.c_str());

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
        this->m_rtss->m_cxt->set_geometry (this->m_img);
    }

    /* Load demographics */
    if (xpd->m_demographic_fn.not_empty()) {
        Xio_demographic demographic ((const char*) xpd->m_demographic_fn);
        if (demographic.m_patient_name.not_empty()) {
            d_ptr->m_meta->set_metadata (0x0010, 0x0010, 
                (const char*) demographic.m_patient_name);
        }
        if (demographic.m_patient_id.not_empty()) {
            d_ptr->m_meta->set_metadata (0x0010, 0x0020, 
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
        if (m_rdd->m_loaded) {
            /* Determine transformation based on original DICOM */
            d_ptr->m_xio_transform->set_from_rdd (this->m_img, 
                d_ptr->m_meta, rdd);
        }
    }

    if (this->m_img) {
        xio_ct_apply_transform (this->m_img, d_ptr->m_xio_transform);
    }
    if (this->m_rtss->m_cxt) {
        xio_structures_apply_transform (this->m_rtss->m_cxt, 
            d_ptr->m_xio_transform);
    }
    if (this->m_dose) {
        xio_dose_apply_transform (this->m_dose, d_ptr->m_xio_transform);
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
    m_rdd->load (rdd);

    if (m_rdd->m_loaded) {
        /* Default to patient position in referenced DICOM */
        d_ptr->m_meta->set_metadata(0x0018, 0x5100,
            m_rdd->m_demographics.get_metadata(0x0018, 0x5100));
        d_ptr->m_xio_transform->set (d_ptr->m_meta);

        /* Default to patient name/ID/sex in referenced DICOM */
        d_ptr->m_meta->set_metadata(0x0010, 0x0010,
            m_rdd->m_demographics.get_metadata(0x0010, 0x0010));
        d_ptr->m_meta->set_metadata(0x0010, 0x0020,
            m_rdd->m_demographics.get_metadata(0x0010, 0x0020));
        d_ptr->m_meta->set_metadata(0x0010, 0x0040,
            m_rdd->m_demographics.get_metadata(0x0010, 0x0040));
    }
}

void
Rtds::load_dose_xio (const char *dose_xio)
{
    if (this->m_dose) {
        delete this->m_dose;
    }
    if (dose_xio) {
        d_ptr->m_xio_dose_filename = dose_xio;
        this->m_dose = new Plm_image ();
        xio_dose_load (this->m_dose, d_ptr->m_meta, dose_xio);
        xio_dose_apply_transform (this->m_dose, d_ptr->m_xio_transform);
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
        astroid_dose_load (this->m_dose, d_ptr->m_meta, dose_astroid);
        astroid_dose_apply_transform (this->m_dose, d_ptr->m_xio_transform);
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
        mc_dose_apply_transform (this->m_dose, d_ptr->m_xio_transform);
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
Rtds::save_dicom_dose (const char *dicom_dir)
{
    if (!dicom_dir) {
        return;
    }

#if PLM_DCM_USE_DCMTK
    this->save_dcmtk_dose (dicom_dir);
#else
    /* Not yet supported -- this function is only used by topas, 
       which uses dcmtk. */
#endif
}

void 
Rtds::set_user_metadata (std::vector<std::string>& metadata)
{
    Metadata *study_metadata = d_ptr->m_drs->get_study_metadata ();

    std::vector<std::string>::iterator it = metadata.begin();
    while (it < metadata.end()) {
        const std::string& str = (*it);
        size_t eq_pos = str.find_first_of ('=');
        if (eq_pos != std::string::npos) {
            std::string key = str.substr (0, eq_pos);
            std::string val = str.substr (eq_pos+1);
            /* Set older-style metadata, used by gdcm */
            d_ptr->m_meta->set_metadata (key, val);
            /* Set newer-style metadata, used by dcmtk */
            study_metadata->set_metadata (key, val);
        }
        ++it;
    }

    d_ptr->m_xio_transform->set (d_ptr->m_meta);
}

void 
Rtds::set_dose (Plm_image *pli)
{
    if (m_dose) delete m_dose;
    m_dose = pli;
}

void 
Rtds::set_dose (Volume *vol)
{
    if (!vol) return;
    if (m_dose) delete m_dose;
    m_dose = new Plm_image;
    /* Make a copy */
    this->m_dose->set_gpuit (vol->clone());
}

Xio_ct_transform*
Rtds::get_xio_ct_transform ()
{
    return d_ptr->m_xio_transform;
}

const std::string&
Rtds::get_xio_dose_filename (void) const
{
    return d_ptr->m_xio_dose_filename;
}

Metadata*
Rtds::get_metadata (void)
{
    return d_ptr->m_meta;
}

Volume*
Rtds::get_volume_short (void)
{
    if (!this->m_img) {
        return 0;
    }
    return this->m_img->get_volume_short ();
}

Volume*
Rtds::get_volume_float (void)
{
    if (!this->m_img) {
        return 0;
    }
    return this->m_img->get_volume_float ();
}
