/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _registration_data_h_
#define _registration_data_h_

#include "plm_config.h"
#include "itk_image.h"
#include "pointset.h"

class Plm_image;
class Registration_parms;

class Registration_data {
public:
    /* Input images */
    Plm_image *fixed_image;
    Plm_image *moving_image;
    UCharImageType::Pointer fixed_mask;
    UCharImageType::Pointer moving_mask;

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
    void load_input_files (Registration_parms* regp);
};

#endif
