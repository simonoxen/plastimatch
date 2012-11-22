/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _contour_distance_h_
#define _contour_distance_h_

#include "plmutil_config.h"
#include "itk_image_type.h"

class Plm_image;
class Contour_distance_private;

/*! \brief 
 * The Contour_distance class computes the distance between 
 * two region boundaries.
 *
 * If the images do not have the same size and resolution, the compare 
 * image will be resampled onto the reference image geometry prior 
 * to comparison.  
 */
class PLMUTIL_API Contour_distance {
public:
    Contour_distance ();
    ~Contour_distance ();
public:
    Contour_distance_private *d_ptr;
public:

    /*! \name Inputs */
    ///@{
    /*! \brief Set the reference image.  The image will be loaded
      from the specified filename. */
    void set_reference_image (const char* image_fn);
    /*! \brief Set the reference image as an ITK image. */
    void set_reference_image (const UCharImageType::Pointer image);
    /*! \brief Set the compare image.  The image will be loaded
      from the specified filename. */
    void set_compare_image (const char* image_fn);
    /*! \brief Set the compare image as an ITK image. */
    void set_compare_image (const UCharImageType::Pointer image);
    ///@}

    /*! \name Execution */
    ///@{
    /*! \brief Compute hausdorff distances */
    void run ();
    ///@}

    /*! \name Outputs */
    ///@{
    /*! \brief Return the Contour distance */
    float get_mean_distance ();
    /*! \brief Display debugging information to stdout */
    void debug ();
    ///@}
};

PLMUTIL_API
void do_contour_mean_distance (
    UCharImageType::Pointer image_1, 
    UCharImageType::Pointer image_2);

#endif
