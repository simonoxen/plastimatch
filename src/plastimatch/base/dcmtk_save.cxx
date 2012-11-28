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
#include "dcmtk_save_p.h"
#include "dcmtk_series.h"
#include "dcmtk_uid.h"
#include "dicom_rt_study.h"
#include "plm_image.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "volume.h"

Dcmtk_save::Dcmtk_save ()
{
    this->d_ptr = new Dcmtk_save_private ();
    this->cxt = 0;
    this->dose = 0;
    this->img = 0;
}

Dcmtk_save::~Dcmtk_save ()
{
    delete d_ptr;

    /* Do nothing with other items -- we don't own the data */
}

void 
Dcmtk_save::set_rt_study (Dicom_rt_study *drs)
{
    d_ptr->m_drs = drs;
}

void Dcmtk_save::set_cxt (Rtss_structure_set *cxt)
{
    this->cxt = cxt;
}

void Dcmtk_save::set_dose (Volume *vol)
{
    this->dose = vol;
}

void Dcmtk_save::set_image (Plm_image* img)
{
    this->img = img;
}

void
Dcmtk_save::save (const char *dicom_dir)
{
    if (this->img) {
        this->save_image (dicom_dir);
    }
    if (this->cxt) {
        this->save_rtss (dicom_dir);
    }
    if (this->dose) {
        this->save_dose (dicom_dir);
    }
}
