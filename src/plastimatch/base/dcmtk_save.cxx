/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_image.h"
#include "dcmtk_rtss.h"
#include "dcmtk_save.h"
#include "dcmtk_series.h"
#include "dcmtk_uid.h"
#include "plm_image.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "volume.h"

Dcmtk_save::Dcmtk_save ()
{
    this->cxt = 0;
    this->cxt_meta = 0;
    this->dose = 0;
    this->img = 0;
}

Dcmtk_save::~Dcmtk_save ()
{
    /* Do nothing, we don't own the data */
}

void Dcmtk_save::set_cxt (Rtss_structure_set *cxt, Metadata *cxt_meta)
{
    this->cxt = cxt;
    this->cxt_meta = cxt_meta;
}

void Dcmtk_save::set_dose (Plm_image* img)
{
    this->dose = img;
}

void Dcmtk_save::set_dose (Volume *vol)
{
    this->dose = new Plm_image ();
    this->dose->set_gpuit (vol);
}

void Dcmtk_save::set_image (Plm_image* img)
{
    this->img = img;
}

void
Dcmtk_save::save (const char *dicom_dir)
{
    Dcmtk_study_writer dsw;
    DcmDate::getCurrentDate (dsw.date_string);
    DcmTime::getCurrentTime (dsw.time_string);
    dcmtk_uid (dsw.study_uid, PLM_UID_PREFIX);
    dcmtk_uid (dsw.for_uid, PLM_UID_PREFIX);
    dcmtk_uid (dsw.ct_series_uid, PLM_UID_PREFIX);
    dcmtk_uid (dsw.rtss_series_uid, PLM_UID_PREFIX);
    dcmtk_uid (dsw.rtss_instance_uid, PLM_UID_PREFIX);

    if (this->img) {
        this->save_image (&dsw, dicom_dir);
    }
    if (this->cxt) {
        this->save_rtss (&dsw, dicom_dir);
    }
}
