/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _registration_data_h_
#define _registration_data_h_

#include "plmregister_config.h"
#include "itk_image_type.h"
#include "plm_image.h"
#include "pointset.h"

class Plm_image;
class Registration_parms;
class Shared_parms;
class Stage_parms;

class PLMREGISTER_API Registration_data {
public:
    /* Input images */
    Plm_image::Pointer fixed_image;
    Plm_image::Pointer moving_image;
    Plm_image::Pointer fixed_roi;
    Plm_image::Pointer moving_roi;

    /* Input landmarks */
    Labeled_pointset *fixed_landmarks;
    Labeled_pointset *moving_landmarks;

    /* Region of interest */
    FloatImageType::RegionType fixed_region;
    FloatImageType::PointType fixed_region_origin;
    FloatImageType::SpacingType fixed_region_spacing;
public:
    Registration_data ();
    ~Registration_data ();
    void load_global_input_files (Registration_parms* regp);
    void load_stage_input_files (Stage_parms* regp);
    void load_shared_input_files (const Shared_parms* shared);
};

#endif
