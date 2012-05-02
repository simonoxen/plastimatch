/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _landmark_diff_h_
#define _landmark_diff_h_

#include "plm_config.h"

typedef struct raw_pointset Raw_pointset;


#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
int
landmark_diff (
    Raw_pointset *rps0,
    Raw_pointset *rps1
);


#if defined __cplusplus
}
#endif

#endif /* _landmark_diff_h_ */
