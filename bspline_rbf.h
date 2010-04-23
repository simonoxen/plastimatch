/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_rbf_h_
#define _bspline_rbf_h_

#include "plm_config.h"

#if defined __cplusplus
extern "C" {
#endif

void bspline_rbf_find_coeffs( Volume *vector_field, BSPLINE_Parms *parms );

void bspline_rbf_update_vector_field( Volume *vector_field,	BSPLINE_Parms *parms );

#if defined __cplusplus
}
#endif

#endif
