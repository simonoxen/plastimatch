/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _distance_map_h_
#define _distance_map_h_

#include "plmutil_config.h"
#include "itk_image_type.h"

class Distance_map_private;
class Plm_image;

class PLMUTIL_API Distance_map {
public:
    Distance_map ();
    ~Distance_map ();
public:
    Distance_map_private *d_ptr;

public:
    /*! \name Inputs */
    ///@{
    /*! \brief Set the input image.  The image will be loaded from 
      the specified filename as a binary mask image (unsigned char). */
    void set_input_image (const char* image_fn);
    /*! \brief Set the input image as an ITK image. */
    void set_input_image (const UCharImageType::Pointer image);
    ///@}

    /*! \name Execution */
    ///@{
    /*! \brief Compute gamma value at each location in the input image */
    void run ();
    ///@}

    /*! \name Outputs */
    ///@{
    /*! \brief Return the gamma image as an ITK image.  */
    FloatImageType::Pointer get_output_image ();
    ///@}
};

#endif
