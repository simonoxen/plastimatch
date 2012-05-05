/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dvh_h_
#define _dvh_h_

#include "plmutil_config.h"
#include <string>
#include "itk_image_type.h"
#include "pstring.h"

class Plm_image;

enum Dvh_units {
    DVH_UNITS_GY,
    DVH_UNITS_CGY,
};

enum Dvh_normalization {
    DVH_NORMALIZATION_PCT,
    DVH_NORMALIZATION_VOX,
};

class API Dvh_parms {
public:
    enum Dvh_units input_units;
    enum Dvh_normalization normalization;
    int cumulative;
    int num_bins;
    float bin_width;
public:
    Dvh_parms ();
};

class API Dvh_parms_pcmd {
public:
    Pstring input_ss_img_fn;
    Pstring input_ss_list_fn;
    Pstring input_dose_fn;
    Pstring output_csv_fn;
    Dvh_parms dvh_parms;
public:
    Dvh_parms_pcmd () {
        printf ("Dvh_parms_pcmd\n");
    }
};

API std::string dvh_execute (
    Plm_image *input_ss_img,
    Plm_image *input_dose_img,
    Dvh_parms *parms);

API std::string dvh_execute (
    UInt32ImageType::Pointer input_ss_img,
    FloatImageType::Pointer input_dose_img,
    Dvh_parms *parms);

API void dvh_execute (
    Dvh_parms_pcmd *dvh_parms_pcmd);

#endif
