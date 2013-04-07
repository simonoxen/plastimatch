/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_image_convert_h_
#define _plm_image_convert_h_

#include "plmbase_config.h"

class Plm_image;

template<class T, class U> PLMBASE_API T plm_image_convert_gpuit_to_itk (
        Plm_image* pli,
        T itk_img, U
);
template<class T, class U> PLMBASE_API void plm_image_convert_itk_to_gpuit (
        Plm_image* pli,
        T img,
        U
);
template<class T> PLMBASE_API void plm_image_convert_itk_to_gpuit_float (
        Plm_image* pli,
        T img
);

#if defined (commentout)
#endif

#endif
