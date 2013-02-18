/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _hausdorff_distance_h_
#define _hausdorff_distance_h_

#include "plmutil_config.h"
#include "itk_image.h"

class Plm_image;
class Hausdorff_distance_private;

/*! \brief 
 * The Hausdorff class computes the worst-case distance between 
 * two regions.
 *
 * If the images do not have the same size and resolution, the compare 
 * image will be resampled onto the reference image geometry prior 
 * to comparison.  
 */
class PLMUTIL_API Hausdorff_distance {
public:
    Hausdorff_distance ();
    ~Hausdorff_distance ();
public:
    Hausdorff_distance_private *d_ptr;
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
    /*! \brief Compute hausdorff distances (obsolete version, doesn't 
      compute 95% Hausdorff) */
    void run_obsolete ();
    ///@}

    /*! \name Outputs */
    ///@{
    /*! \brief Return the Hausdorff distance */
    float get_hausdorff ();
    /*! \brief Return the average Hausdorff distance */
    float get_average_hausdorff ();
    /*! \brief Display debugging information to stdout */
    void debug ();
    ///@}

protected:
    void run_internal (
        UCharImageType::Pointer image,
        FloatImageType::Pointer dmap);
};

PLMUTIL_API
void do_hausdorff(
    UCharImageType::Pointer image_1, 
    UCharImageType::Pointer image_2);

#endif
