/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"

#include "plmutil.h"

#include "compiler_warnings.h"
#include "dcmtk_loader.h"
#include "dcmtk_save.h"

void
Rtds::load_dcmtk (const char *dicom_dir)
{
#if PLM_DCM_USE_DCMTK
    Dcmtk_series_set dss (dicom_dir);
    Dcmtk_series_set::Dcmtk_series_map::iterator it;
    Dcmtk_series *ds_rtdose = 0;
    Dcmtk_series *ds_rtss = 0;

    /* First pass: loop through series and find ss, dose */
    /* GCS FIX: maybe need additional pass, make sure ss & dose 
       refer to same CT, in case of multiple ss & dose in same 
       directory */
    for (it = dss.m_smap.begin(); it != dss.m_smap.end(); ++it) {
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
    for (it = dss.m_smap.begin(); it != dss.m_smap.end(); ++it) {
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
	    this->m_img = ds->load_plm_image ();
	    continue;
	}
    }

#if defined (GCS_FIX)
    if (ds_rtss) {
        ds_rtss->rtss_load (rtds);
    }

    if (ds_rtdose) {
        ds_rtdose->rtdose_load (rtds);
    }
#endif

    printf ("Done.\n");
#endif
}

void
Rtds::save_dcmtk (const char *dicom_dir)
{
#if PLM_DCM_USE_DCMTK
    dcmtk_rtds_save (this, dicom_dir);
#endif
}
