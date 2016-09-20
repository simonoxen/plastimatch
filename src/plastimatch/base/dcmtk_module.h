/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_module_h_
#define _dcmtk_module_h_

#include "plmbase_config.h"
#include <string>
#include "rt_study_metadata.h"

class DcmDataset;

class PLMBASE_API Dcmtk_module_patient {
public:
    static void set (DcmDataset *dataset, const Metadata::Pointer& meta);
};

class PLMBASE_API Dcmtk_module_general_study {
public:
    static void set (DcmDataset *dataset, 
        const Rt_study_metadata::Pointer& rsm);
};

class PLMBASE_API Dcmtk_module_general_series {
public:
    static void set_sro (DcmDataset *dataset, 
        const Rt_study_metadata::Pointer& rsm);
};

#endif
