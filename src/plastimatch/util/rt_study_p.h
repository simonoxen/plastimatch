/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rt_study_p_h_
#define _rt_study_p_h_

#include "plmutil_config.h"
#include "metadata.h"
#include "rt_study_metadata.h"
#include "segmentation.h"
#include "slice_index.h"
#include "xio_ct_transform.h"

class PLMUTIL_API Rt_study_private {
public:
    Rt_study_metadata::Pointer m_drs;     /* UIDs, etc -- used by dcmtk */
    Metadata *m_meta;                  /* Patient name, patient id, etc. */
    std::string m_xio_dose_filename;   /* XiO dose file to use as template 
                                          for saving in XiO format */
    Xio_ct_transform *m_xio_transform; /* Transformation from XiO to DICOM
                                          coordinates */
    Slice_index *m_slice_index;        /* UIDs, etc -- used by gdcm */

    Plm_image::Pointer m_img;          /* CT image */
    Plm_image::Pointer m_dose;         /* RT dose */
    Segmentation::Pointer m_rtss;      /* RT structure set */

public:
    Rt_study_private () {
        m_drs = Rt_study_metadata::New ();
        m_meta = new Metadata;
        m_meta->create_anonymous ();
        m_xio_transform = new Xio_ct_transform (this->m_meta);
        m_slice_index = new Slice_index;
    }
    ~Rt_study_private () {
        delete m_meta;
        delete m_slice_index;
        delete m_xio_transform;
    }
};

#endif
