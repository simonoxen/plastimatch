/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gamma_dose_comparison_h_
#define _gamma_dose_comparison_h_

#include "plmutil_config.h"
#include "plm_macros.h"
#include "itk_image_type.h"

class Gamma_dose_comparison_private;
class Plm_image;

/*! \brief 
 * The Gamma_dose_comparison class executes a comparison between 
 * two dose distributions based on the "gamma index" defined by 
 * Dan Low et al. in the following reference:
 * \n\n
 *   Low et al, 
 *   A technique for the quantitative evaluation of dose distributions,
 *   Med Phys. 1998 May;25(5):656-61.
 * \n\n
 * The comparison is based on searching a local neighborhood for the 
 * most similar dose.  The similarity is computed as the geometric mean 
 * of the dose difference and spatial distance.  The gamma value at 
 * a point is then the minumum of this similarity value over the 
 * the neighborhood.
 * Generally, the gamma value is normalized based on a spatial tolerance
 * and a dose difference tolerance such that gamma values of less than 
 * 1.0 are acceptable, and gamma values greater than 1.0 are unacceptable.
 */
class PLMUTIL_API Gamma_dose_comparison {
public:
    Gamma_dose_comparison ();
    ~Gamma_dose_comparison ();
public:
    Gamma_dose_comparison_private *d_ptr;
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
    /*! \brief Set an absolute dose threshold for gamma analysis, 
      in Gray.  This is used to ignore voxels which have dose 
      below a certain value.  
      For example, you may wish to consider only voxels which 
      have dose greater than 10% of the prescription dose.
      If the prescription dose is 60 Gy, you would 
      call set_analysis_threshold_abs (6.0).
      The threshold is applied to the reference image. 
    */
    void set_analysis_threshold_abs (float abs_thresh);
    /*! \brief Set a dose threshold for gamma analysis, as a 
      percent of the maximum dose.  This is used to ignore 
      voxels which have dose below a certain value.  
      For example, to consider only voxels which have dose greater 
      than 10% of the maximum dose, you would call
      set_analysis_threshold_pct_max (0.1).
      The threshold is applied to the reference image. 
    */
    void set_analysis_threshold_pct_max (float pct_thresh);
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
