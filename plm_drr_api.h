/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _drr_api_h_
#define _drr_api_h_

#ifdef PLM_INTERNAL

/* Internal definition of API */
#include "plm_config.h"
#include "itk_image.h"

typedef struct plm_drr_context* Plm_drr_context;
struct plm_drr_context {
    FloatImageType *moving;
    FloatImageType *fixed;
    char *command_string;
    int status;
};

#else /* PLM_INTERNAL */

/* External users of API */
# ifdef plastimatch1_EXPORTS
#  define plastimatch1_EXPORT __declspec(dllexport)
# else
#  define plastimatch1_EXPORT __declspec(dllimport)
# endif

typedef (void*) Plm_drr_context;

#endif /* PLM_INTERNAL */

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
Plm_drr_context plm_drr_context_create ();
plastimatch1_EXPORT
void plm_drr_context_destroy (Plm_drr_context);

#if defined __cplusplus
}
#endif

#endif
