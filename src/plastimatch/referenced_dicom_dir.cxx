/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>

#include "dcm_util.h"
#include "math_util.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "print_and_exit.h"
#include "referenced_dicom_dir.h"
#include "rtss_polyline_set.h"

Referenced_dicom_dir::Referenced_dicom_dir ()
{
    this->m_loaded = 0;
}

Referenced_dicom_dir::~Referenced_dicom_dir ()
{
}

void
Referenced_dicom_dir::load (const char *dicom_dir)
{
    dcm_load_rdd (this, dicom_dir);
}

void
Referenced_dicom_dir::get_slice_info (
    int *slice_no,                  /* Output */
    Pstring *ct_slice_uid,          /* Output */
    float z                         /* Input */
) const
{
    if (!this->m_loaded) {
	*slice_no = -1;
	return;
    }

    /* NOTE: This algorithm doesn't work if there are duplicate slices */
    *slice_no = ROUND_INT ((z - this->m_pih.m_origin[2]) 
	/ this->m_pih.m_spacing[2]);
    if (*slice_no < 0 || *slice_no >= this->m_pih.Size(2)) {
	*slice_no = -1;
	return;
    }

    (*ct_slice_uid) = this->m_ct_slice_uids[*slice_no];
}
