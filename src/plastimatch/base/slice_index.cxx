/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdio.h>
#include <stdlib.h>

#include "plmbase.h"
#include "plm_math.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"


Slice_index::Slice_index ()
{
    this->m_loaded = 0;
}

Slice_index::~Slice_index ()
{
}

void
Slice_index::load (const char *dicom_dir)
{
    dcm_load_rdd (this, dicom_dir);
}

void
Slice_index::get_slice_info (
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
