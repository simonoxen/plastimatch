/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dvh_p_h_
#define _dvh_p_h_

#include "plmutil_config.h"
#include <string>
#include "itk_image_type.h"
#include "dvh.h"

class Dvh_private {
public:
    Dvh_private ();
    ~Dvh_private ();
public:
    Rtss *rtss;
    Plm_image *dose;
    enum Dvh::Dvh_units dose_units;
    enum Dvh::Dvh_normalization normalization;
    enum Dvh::Histogram_type histogram_type;
    int num_bins;
    float bin_width;
    std::string output_string;
};

#endif
