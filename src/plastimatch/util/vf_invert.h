/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _vf_invert_h_
#define _vf_invert_h_

#include "plmutil_config.h"
#include "itk_image_type.h"

class Vf_invert_private;

class PLMUTIL_API Vf_invert {
public:
    Vf_invert ();
    ~Vf_invert ();
public:
    Vf_invert_private *d_ptr;
public:

    /*! \name Inputs */
    ///@{
    /*! \brief Set the input vector field to be inverted.  
      The vector field will be loaded from the specified filename. */
    void set_input_vf (const char* vf_fn);
    /*! \brief Set the input vector field to be inverted
      as an ITK image. */
    void set_input_vf (const DeformationFieldType::Pointer vf);
    /*! \brief Set the geometry of the output vector field to match 
      another image.
      The image will be loaded from the specified filename. */
    void set_fixed_image (const char* image_fn);
    /*! \brief Set the geometry of the output vector field to match 
      another image, specified as an ITK image. */
    void set_fixed_image (const FloatImageType::Pointer image);
    ///@}

    /*! \name Execution */
    ///@{
    /*! \brief Compute inverse vector field */
    void run ();
    ///@}
};

#endif
