/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _sift_h_
#define _sift_h_

#include "plmutil_config.h"
#include "plm_macros.h"
#include "itk_image_type.h"

class Sift_private;
class Plm_image;

/*! \brief 
 * The Sift class implements a SIFT feature detector.
 */
class PLMUTIL_API Sift {
public:
    Sift ();
    ~Sift ();
public:
    Sift_private *d_ptr;
public:

    /*! \name Inputs */
    ///@{
    /*! \brief Set the reference image.  The image will be loaded
      from the specified filename. */
    void set_reference_image (const char* image_fn);
    /*! \brief Set the reference image as a Plm image. */
    void set_reference_image (Plm_image* image);
    /*! \brief Set the reference image as an ITK image. */
    void set_reference_image (const FloatImageType::Pointer image);
    /*! \brief Set the compare image.  The image will be loaded
      from the specified filename. */
    void set_compare_image (const char* image_fn);
    /*! \brief Set the compare image as a Plm image. */
    void set_compare_image (Plm_image* image);
    /*! \brief Set the compare image as an ITK image. */
    void set_compare_image (const FloatImageType::Pointer image);

    /*! \brief Get the distance to agreement (DTA) tolerance, in mm. */
    float get_spatial_tolerance ();
    /*! \brief Set the distance to agreement (DTA) tolerance, in mm. */
    void set_spatial_tolerance (float spatial_tol);
    /*! \brief Get the dose difference tolerance, in percent. */
    float get_dose_difference_tolerance ();
    /*! \brief Set the dose difference tolerance, in percent. 
      If a reference dose (prescription dose) is specified, 
      the dose difference tolerance is treated as a 
      percent of the reference dose.  Otherwise it is treated as a
      percent of the maximum dose in the reference volume.  
      To use a 3% dose tolerance, you would set this value to 0.03.  */
    void set_dose_difference_tolerance (float dose_tol);
    /*! \brief Set the reference dose (prescription dose).  This 
      is used in dose comparison. */
    void set_reference_dose (float dose);
    /*! \brief Set the dose threshold for gamma analysis.  
      This is used to ignore voxels which have dose below a certain value.  
      For example, to consider only voxels which have dose greater 
      than 10% of the prescription dose, set the analysis threshold to 
      0.10.  The threshold is applied to the reference image. 
      <b>Not yet implemented - threshold is hard-coded</b> */
    void set_analysis_threshold (float percent);
    /*! \brief Set the maximum gamma computed by the class.  This is 
      used to speed up computation.  A typical value is 2 or 3.  */
    void set_gamma_max (float gamma_max);
    ///@}

    /*! \name Execution */
    ///@{
    /*! \brief Compute gamma value at each location in the input image */
    void run ();
    ///@}

    /*! \name Outputs */
    ///@{
    /*! \brief Return the gamma image as a Plm_image.  */
    Plm_image* get_gamma_image ();
    /*! \brief Return the gamma image as an ITK image.  */
    FloatImageType::Pointer get_gamma_image_itk ();
    /*! \brief Return a binary image of passing voxels as a Plm image. */
    Plm_image* get_pass_image ();
    /*! \brief Return a binary image of passing voxels as an ITK image. */
    UCharImageType::Pointer get_pass_image_itk ();
    /*! \brief Return a binary image of failing voxels as a Plm image. */
    Plm_image* get_fail_image ();
    /*! \brief Return a binary image of failing voxels as an ITK image. */
    UCharImageType::Pointer get_fail_image_itk ();
    ///@}
    /*! \brief Resample image_moving to image_reference */
    void resample_image_to_reference (Plm_image *image_reference, Plm_image *image_moving);
};

#endif
