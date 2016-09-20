/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_module_h_
#define _dcmtk_module_h_

#include "plmbase_config.h"
#include <string>
#include "rt_study_metadata.h"

class DcmDataset;

class PLMBASE_API Dcmtk_module {
public:
    static void set_patient (
        DcmDataset *dataset,
        const Metadata::Pointer& meta);
    static void set_general_study (
        DcmDataset *dataset, 
        const Rt_study_metadata::Pointer& rsm);
    static void set_general_series_sro (
        DcmDataset *dataset, 
        const Rt_study_metadata::Pointer& rsm);
    static void set_rt_series (
        DcmDataset *dataset, 
        const Metadata::Pointer& rsm,
        const char* modality);
};

#endif
