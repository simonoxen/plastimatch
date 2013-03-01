/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_p_h_
#define _rtds_p_h_

#include "plmutil_config.h"
#include "dicom_rt_study.h"
#include "metadata.h"
#include "xio_ct_transform.h"

class PLMUTIL_API Rtds_private {
public:
    Dicom_rt_study *m_drs;             /* UIDs, etc -- used by dcmtk */
    Metadata *m_meta;                  /* Patient name, patient id, etc. */
    std::string m_xio_dose_filename;   /* XiO dose file to use as template 
                                          for saving in XiO format */
    Xio_ct_transform *m_xio_transform; /* Transformation from XiO to DICOM
                                          coordinates */
public:
    Rtds_private () {
        m_drs = new Dicom_rt_study;
        m_meta = new Metadata;
        m_meta->create_anonymous ();
        m_xio_transform = new Xio_ct_transform (this->m_meta);
    }
    ~Rtds_private () {
        delete m_drs;
        delete m_meta;
        delete m_xio_transform;
    }
};

#endif
