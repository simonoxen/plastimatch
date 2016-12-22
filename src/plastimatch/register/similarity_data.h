/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _similarity_data_h_
#define _similarity_data_h_

#include "plmregister_config.h"

class PLMREGISTER_API Similarity_data
{
public:
    SMART_POINTER_SUPPORT (Similarity_data);
public:
    Plm_image::Pointer fixed;
    Plm_image::Pointer moving;
    Plm_image::Pointer fixed_grad;
    Plm_image::Pointer moving_grad;
    Plm_image::Pointer fixed_roi;
    Plm_image::Pointer moving_roi;
};

#endif
