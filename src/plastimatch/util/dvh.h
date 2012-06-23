/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dvh_h_
#define _dvh_h_

#include "plmutil_config.h"
#include <string>
#include "itk_image_type.h"
#include "pstring.h"

#if defined (commentout)
class Plm_image;

enum Dvh_units {
    DVH_UNITS_GY,
    DVH_UNITS_CGY,
};

enum Dvh_normalization {
    DVH_NORMALIZATION_PCT,
    DVH_NORMALIZATION_VOX,
};

class PLMUTIL_API Dvh_parms {
public:
    enum Dvh_units input_units;
    enum Dvh_normalization normalization;
    int cumulative;
    int num_bins;
    float bin_width;
public:
    Dvh_parms ();
};

class PLMUTIL_API Dvh_parms_pcmd {
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

PLMUTIL_API std::string dvh_execute (
    Plm_image *input_ss_img,
    Plm_image *input_dose_img,
    Dvh_parms *parms);

PLMUTIL_API std::string dvh_execute (
    UInt32ImageType::Pointer input_ss_img,
    FloatImageType::Pointer input_dose_img,
    Dvh_parms *parms);

PLMUTIL_API void dvh_execute (
    Dvh_parms_pcmd *dvh_parms_pcmd);
#endif

class Dvh_private;
class Plm_image;
class Rtds;
class Rtss;

class Dvh {
public:
    Dvh ();
    ~Dvh ();
public:
    Dvh_private *d_ptr;

public:
    enum Dvh_units {
        DVH_UNITS_GY,
        DVH_UNITS_CGY,
    };

    enum Dvh_normalization {
        DVH_NORMALIZATION_PCT,
        DVH_NORMALIZATION_VOX,
    };

public:

    /*! \name Inputs */
    ///@{
    /*! \brief Set the structure set image.  The image will be loaded
      from the specified filename, and an optional file containing the 
      image list will be loaded. */
    void set_structure_set_image (const char* ss_image_fn, 
        const char *ss_list_fn);
    /*! \brief Set the structure set image as an Rtss */
    void set_structure_set_image (Rtss* image);
    /*! \brief Set the dose image.  The image will be loaded
      from the specified filename. */
    void set_dose_image (const char* image_fn);
    /*! \brief Set the dose image as a Plm image. */
    void set_dose_image (Plm_image* image);
    /*! \brief Set the dose image as an ITK image. */
    void set_dose_image (const FloatImageType::Pointer image);

    /*! \brief Set the units for dose image. */
    void set_dose_units (enum Dvh_units units);
    /*! \brief Set the units for dvh computation.  Normalization in 
      either percent or voxels, choice of cumulative or differential 
      histogram, number of bins, and bin width. */
    void set_dvh_parameters (enum Dvh_normalization normalization,
        int cumulative, int num_bins, float bin_width);

    ///@}

    /*! \name Execution */
    ///@{
    /*! \brief Compute dvh */
    void run ();
    ///@}

    /*! \name Outputs */
    ///@{
    /*! \brief Save the DVH as a csv file */
    void save_csv (const char* csv_fn);
    ///@}

};

#endif
