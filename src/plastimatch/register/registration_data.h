/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _registration_data_h_
#define _registration_data_h_

#include "plmregister_config.h"
#include "itk_image_type.h"
#include "pointset.h"

//template<class T> class PLMREGISTER_API Pointset;

class Plm_image;
class Registration_parms;

class PLMREGISTER_API Registration_data {
public:
    /* Input images */
    Plm_image *fixed_image;
    Plm_image *moving_image;
    Plm_image *fixed_roi;
    Plm_image *moving_roi;

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
