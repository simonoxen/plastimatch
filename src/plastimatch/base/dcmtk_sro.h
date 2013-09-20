/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_sro_h_
#define _dcmtk_sro_h_

#include "plmbase_config.h"
#include <string>
#include "rt_study_metadata.h"

class Xform;

class PLMBASE_API Dcmtk_sro {
public:
    static void save (
        Xform* xf,
        const Rt_study_metadata::Pointer& rtm_src,
        const Rt_study_metadata::Pointer& rtm_reg,
        const std::string& dicom_dir);
};

#endif
