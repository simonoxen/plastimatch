/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_p_h_
#define _rtds_p_h_

#include "plmutil_config.h"
#include "dicom_rt_study.h"
#include "metadata.h"
#include "rtss.h"
#include "slice_index.h"
#include "xio_ct_transform.h"

class PLMUTIL_API Rtds_private {
public:
    Dicom_rt_study::Pointer m_drs;     /* UIDs, etc -- used by dcmtk */
    Metadata *m_meta;                  /* Patient name, patient id, etc. */
    std::string m_xio_dose_filename;   /* XiO dose file to use as template 
                                          for saving in XiO format */
    Xio_ct_transform *m_xio_transform; /* Transformation from XiO to DICOM
                                          coordinates */
    Slice_index *m_slice_index;        /* UIDs, etc -- used by gdcm */

    Plm_image::Pointer m_img;          /* CT image */
    Plm_image::Pointer m_dose;         /* RT dose */
    Rtss::Pointer m_rtss;              /* RT structure set */

public:
    Rtds_private () {
        m_drs = Dicom_rt_study::New ();
        m_meta = new Metadata;
        m_meta->create_anonymous ();
        m_xio_transform = new Xio_ct_transform (this->m_meta);
        m_slice_index = new Slice_index;
    }
    ~Rtds_private () {
        delete m_meta;
        delete m_slice_index;
        delete m_xio_transform;
    }
};

#endif
