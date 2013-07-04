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
#include "rt_study.h"
#include "rt_study_p.h"
#include "rtss.h"
#include "segmentation.h"
#include "volume.h"
#include "xio_ct.h"
#include "xio_ct_transform.h"
#include "xio_demographic.h"
#include "xio_dir.h"
#include "xio_dose.h"
#include "xio_patient.h"
#include "xio_structures.h"

Rt_study::Rt_study ()
{
    d_ptr = new Rt_study_private;
}

Rt_study::~Rt_study ()
{
    delete d_ptr;
}

void
Rt_study::load_image (const char *fn)
{
    d_ptr->m_img = Plm_image::New (new Plm_image (fn));
}

void
Rt_study::load_image (const std::string& fn)
{
    this->load_image (fn.c_str());
}

void
Rt_study::load_dicom_dir (const char *dicom_dir)
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
Rt_study::load_dicom (const char *dicom_dir)
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
Rt_study::load_dicom_rtss (const char *dicom_path)
{
    d_ptr->m_rtss.reset ();
#if PLM_DCM_USE_DCMTK
    this->load_dcmtk (dicom_path);
#elif GDCM_VERSION_1
    d_ptr->m_rtss = Segmentation::New (new Segmentation (this));
    d_ptr->m_rtss->load_gdcm_rtss (dicom_path, d_ptr->m_drs.get());
#else
    /* Do nothing */
#endif
}

void
Rt_study::load_dicom_dose (const char *dicom_path)
{
#if PLM_DCM_USE_DCMTK
    this->load_dcmtk (dicom_path);
#elif GDCM_VERSION_1
    d_ptr->m_dose.reset (gdcm1_dose_load (0, dicom_path));
#else
    /* Do nothing */
#endif
}

void
Rt_study::load_xio (const char *xio_dir)
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
        d_ptr->m_dose = Plm_image::New ();
        printf ("calling xio_dose_load\n");
        d_ptr->m_xio_dose_filename = std::string(xtpd->path) + "/dose.1";
        xio_dose_load (d_ptr->m_dose.get(), 
            //d_ptr->m_meta,
            d_ptr->m_drs->get_dose_metadata (),
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
    d_ptr->m_img = Plm_image::New();
    xio_ct_load (d_ptr->m_img.get(), &xst);

    /* Load the XiO studyset structure set */
    d_ptr->m_rtss = Segmentation::New (new Segmentation (this));
    d_ptr->m_rtss->load_xio (xst);

    /* Apply XiO CT geometry to structures */
    if (d_ptr->m_rtss->have_structure_set()) {
        Rtss *rtss_ss = d_ptr->m_rtss->get_structure_set_raw ();
        rtss_ss->set_geometry (d_ptr->m_img.get());
    }

    /* Load demographics */
    if (xpd->m_demographic_fn.not_empty()) {
        Xio_demographic demographic ((const char*) xpd->m_demographic_fn);
        if (demographic.m_patient_name.not_empty()) {
            d_ptr->m_drs->set_study_metadata (0x0010, 0x0010, 
                (const char*) demographic.m_patient_name);
        }
        if (demographic.m_patient_id.not_empty()) {
            d_ptr->m_drs->set_study_metadata (0x0010, 0x0020, 
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

    if (d_ptr->m_img) {
        if (d_ptr->m_drs->slice_list_complete()) {
            /* Determine transformation based on original DICOM */
            d_ptr->m_xio_transform->set_from_rdd (d_ptr->m_img.get(), 
                d_ptr->m_drs.get());
        }
    }

    if (d_ptr->m_img) {
        xio_ct_apply_transform (d_ptr->m_img.get(), d_ptr->m_xio_transform);
    }
    if (d_ptr->m_rtss->have_structure_set()) {
        xio_structures_apply_transform (d_ptr->m_rtss->get_structure_set_raw(),
            d_ptr->m_xio_transform);
    }
    if (d_ptr->m_dose) {
        xio_dose_apply_transform (d_ptr->m_dose.get(), d_ptr->m_xio_transform);
    }
}

void
Rt_study::load_ss_img (const char *ss_img, const char *ss_list)
{
    d_ptr->m_rtss = Segmentation::New (new Segmentation (this));
    d_ptr->m_rtss->load (ss_img, ss_list);
}

void
Rt_study::load_dose_img (const char *dose_img)
{
    if (d_ptr->m_dose) {
        d_ptr->m_dose.reset();
    }
    if (dose_img) {
        d_ptr->m_dose.reset(plm_image_load_native (dose_img));
    }
}

void
Rt_study::load_rdd (const char *image_directory)
{
    d_ptr->m_drs = Rt_study_metadata::load (image_directory);

    /* GCS FIX: I think the below is not needed any more, but there 
       might be some edge cases, such as converting image with referenced 
       ct, which should copy patient name but not slice uids */
#if defined (commentout)
    Rt_study_metadata *rsm = d_ptr->m_drs.get ();
    Metadata *meta = rsm->get_study_metadata ();
    if (rsm->slice_list_complete()) {
        /* Default to patient position in referenced DICOM */
        d_ptr->m_meta->set_metadata(0x0018, 0x5100,
            si->m_demographics.get_metadata(0x0018, 0x5100));
        d_ptr->m_xio_transform->set (d_ptr->m_meta);

        /* Default to patient name/ID/sex in referenced DICOM */
        d_ptr->m_meta->set_metadata(0x0010, 0x0010,
            si->m_demographics.get_metadata(0x0010, 0x0010));
        d_ptr->m_meta->set_metadata(0x0010, 0x0020,
            si->m_demographics.get_metadata(0x0010, 0x0020));
        d_ptr->m_meta->set_metadata(0x0010, 0x0040,
            si->m_demographics.get_metadata(0x0010, 0x0040));
    }
#endif
}

void
Rt_study::load_dose_xio (const char *dose_xio)
{
    if (d_ptr->m_dose) {
        d_ptr->m_dose.reset();
    }
    if (dose_xio) {
        d_ptr->m_xio_dose_filename = dose_xio;
        d_ptr->m_dose = Plm_image::New ();
        Metadata *dose_meta = d_ptr->m_drs->get_dose_metadata ();
        xio_dose_load (d_ptr->m_dose.get(), dose_meta, dose_xio);
        xio_dose_apply_transform (d_ptr->m_dose.get(), d_ptr->m_xio_transform);
    }
}

void
Rt_study::load_dose_astroid (const char *dose_astroid)
{
    if (d_ptr->m_dose) {
        d_ptr->m_dose.reset();
    }
    if (dose_astroid) {
        d_ptr->m_dose = Plm_image::New ();
        Metadata *dose_meta = d_ptr->m_drs->get_dose_metadata ();
        astroid_dose_load (d_ptr->m_dose.get(), dose_meta, dose_astroid);
        astroid_dose_apply_transform (d_ptr->m_dose.get(), 
            d_ptr->m_xio_transform);
    }
}

void
Rt_study::load_dose_mc (const char *dose_mc)
{
    if (d_ptr->m_dose) {
        d_ptr->m_dose.reset();
    }
    if (dose_mc) {
        d_ptr->m_dose = Plm_image::New ();
        mc_dose_load (d_ptr->m_dose.get(), dose_mc);
        mc_dose_apply_transform (d_ptr->m_dose.get(), d_ptr->m_xio_transform);
    }
}

void 
Rt_study::load_cxt (const char *input_fn)
{
    d_ptr->m_rtss = Segmentation::New (new Segmentation (this));
    d_ptr->m_rtss->load_cxt (input_fn, d_ptr->m_drs.get());
}

void 
Rt_study::load_prefix (const char *input_fn)
{
    d_ptr->m_rtss = Segmentation::New (new Segmentation (this));
    d_ptr->m_rtss->load_prefix (input_fn);
}

void
Rt_study::save_dicom (const char *dicom_dir)
{
    if (!dicom_dir) {
        return;
    }

    if (d_ptr->m_img) {
        d_ptr->m_drs->set_image_header (d_ptr->m_img);
    }
    if (d_ptr->m_rtss) {
        d_ptr->m_rtss->cxt_extract ();
    }

#if PLM_DCM_USE_DCMTK
    this->save_dcmtk (dicom_dir);
#else
    this->save_gdcm (dicom_dir);
#endif
}

void
Rt_study::save_dicom_dose (const char *dicom_dir)
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
Rt_study::save_dose (const char* fname)
{
    if (d_ptr->m_dose) {
        d_ptr->m_dose->save_image (fname);
    }
}

void
Rt_study::save_dose (const char* fname, Plm_image_type image_type)
{
    if (d_ptr->m_dose) {
        d_ptr->m_dose->convert_and_save (fname, image_type);
    }
}

void
Rt_study::save_prefix (
    const std::string& output_prefix, 
    const std::string& extension)
{
    d_ptr->m_rtss->save_prefix (output_prefix, extension);
}

Rt_study_metadata *
Rt_study::get_rt_study_metadata ()
{
    return d_ptr->m_drs.get();
}

void 
Rt_study::set_user_metadata (std::vector<std::string>& metadata)
{
    Metadata *study_metadata = d_ptr->m_drs->get_study_metadata ();

    std::vector<std::string>::iterator it = metadata.begin();
    while (it < metadata.end()) {
        const std::string& str = (*it);
        size_t eq_pos = str.find_first_of ('=');
        if (eq_pos != std::string::npos) {
            std::string key = str.substr (0, eq_pos);
            std::string val = str.substr (eq_pos+1);
#if defined (commentout)
            /* Set older-style metadata, used by gdcm */
            d_ptr->m_meta->set_metadata (key, val);
#endif
            /* Set newer-style metadata, used by dcmtk */
            study_metadata->set_metadata (key, val);
        }
        ++it;
    }

    d_ptr->m_xio_transform->set (d_ptr->m_drs->get_image_metadata());
}

bool
Rt_study::have_image ()
{
    return (bool) d_ptr->m_img;
}

Plm_image::Pointer
Rt_study::get_image ()
{
    return d_ptr->m_img;
}

void 
Rt_study::set_image (ShortImageType::Pointer itk_image)
{
    d_ptr->m_img = Plm_image::New (new Plm_image(itk_image));
}

void 
Rt_study::set_image (FloatImageType::Pointer itk_image)
{
    d_ptr->m_img = Plm_image::New (new Plm_image(itk_image));
}

void 
Rt_study::set_image (Plm_image* pli)
{
    d_ptr->m_img.reset (pli);
}

void 
Rt_study::set_image (Plm_image::Pointer pli)
{
    d_ptr->m_img = pli;
}

bool
Rt_study::have_dose ()
{
    return (bool) d_ptr->m_dose;
}

void 
Rt_study::set_dose (Plm_image *pli)
{
    d_ptr->m_dose.reset (pli);
}

void 
Rt_study::set_dose (FloatImageType::Pointer itk_dose)
{
    d_ptr->m_dose.reset (new Plm_image (itk_dose));
}

void 
Rt_study::set_dose (Volume *vol)
{
    if (!vol) return;
    d_ptr->m_dose = Plm_image::New();

    /* GCS FIX: Make a copy */
    d_ptr->m_dose->set_volume (vol->clone_raw());
}

Plm_image::Pointer
Rt_study::get_dose ()
{
    return d_ptr->m_dose;
}

bool
Rt_study::have_rtss ()
{
    return (bool) d_ptr->m_rtss;
}

Segmentation::Pointer
Rt_study::get_rtss ()
{
    return d_ptr->m_rtss;
}

void 
Rt_study::set_rtss (Segmentation::Pointer rtss)
{
    d_ptr->m_rtss = rtss;
}

void 
Rt_study::add_structure (
    UCharImageType::Pointer itk_image,
    const char *structure_name,
    const char *structure_color)
{
    if (!have_rtss()) {
        d_ptr->m_rtss = Segmentation::New ();
    }
    d_ptr->m_rtss->add_structure (itk_image, structure_name, structure_color);
}

Xio_ct_transform*
Rt_study::get_xio_ct_transform ()
{
    return d_ptr->m_xio_transform;
}

const std::string&
Rt_study::get_xio_dose_filename (void) const
{
    return d_ptr->m_xio_dose_filename;
}

Metadata*
Rt_study::get_metadata (void)
{
    return d_ptr->m_drs->get_study_metadata();
}

Volume*
Rt_study::get_image_volume_short ()
{
    if (!d_ptr->m_img) {
        return 0;
    }
    return d_ptr->m_img->get_volume_short ();
}

Volume*
Rt_study::get_image_volume_float (void)
{
    if (!d_ptr->m_img) {
        return 0;
    }
    return d_ptr->m_img->get_volume_float_raw ();
}

bool
Rt_study::has_dose ()
{
    return (d_ptr->m_dose != 0);
}

Plm_image*
Rt_study::get_dose_plm_image ()
{
    if (!d_ptr->m_dose) {
        return 0;
    }
    return d_ptr->m_dose.get();
}

Volume*
Rt_study::get_dose_volume_float ()
{
    if (!d_ptr->m_dose) {
        return 0;
    }
    return d_ptr->m_dose->get_volume_float_raw ();
}
