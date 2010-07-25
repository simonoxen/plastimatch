/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_rbf_h_
#define _bspline_rbf_h_

#include "plm_config.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void bspline_rbf_find_coeffs( Volume *vector_field, Bspline_parms *parms );

plastimatch1_EXPORT
void bspline_rbf_update_vector_field( Volume *vector_field, Bspline_parms *parms );

#if defined __cplusplus
}
#endif

#endif
