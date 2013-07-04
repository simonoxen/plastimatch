/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_resample_h_
#define _itk_resample_h_

#include "plmbase_config.h"

class Plm_image_header;

template <class T, class U> T
vector_resample_image (T& vf_image, U& ref_image);
template <class T> T
vector_resample_image (T& image, float x_spacing,
        float y_spacing, float z_spacing);
template <class T> T
vector_resample_image (T& vf_image, Plm_image_header* pih);

template <class T> T
resample_image (T& image, const Plm_image_header* pih, 
    float default_val, int interp_lin);
template <class T> T
resample_image (T& image, const Plm_image_header& pih, 
    float default_val, int interp_lin);
template <class T> T
resample_image (T& image, float spacing[3]);

template <class T>
T
subsample_image (T& image, int x_sampling_rate,
        int y_sampling_rate, int z_sampling_rate,
        float default_val);

#endif
