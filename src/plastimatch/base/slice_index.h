/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _slice_index_h_
#define _slice_index_h_

#include "plmbase_config.h"
#include <vector>
#include "metadata.h"
#include "plm_image_header.h"
#include "pstring.h"

class Slice_index {
public:
    /* Set this if we have m_pih && ct slice uids */
    bool m_loaded;

    Plm_image_header m_pih;
    Metadata m_demographics;
    Pstring m_study_id;
    Pstring m_ct_study_uid;
    Pstring m_ct_series_uid;
    Pstring m_ct_fref_uid;

    /* These must be sorted in order, starting with origin slice */
    std::vector<Pstring> m_ct_slice_uids;

public:
    plastimatch1_EXPORT
    Slice_index ();
    plastimatch1_EXPORT
    ~Slice_index ();
    void load (const char *dicom_dir);
    void get_slice_info (int *slice_no, Pstring *ct_slice_uid, float z) const;

};

#if defined __cplusplus
extern "C" {
#endif

#if defined __cplusplus
}
#endif

#endif
