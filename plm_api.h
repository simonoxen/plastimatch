/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_api_h_
#define _plm_api_h_

#ifdef PLM_INTERNAL
/* Internal definition of API */
#include "plm_config.h"
#include "itk_image.h"

typedef struct plm_registration_context* Plm_registration_context;
struct plm_registration_context {
    FloatImageType *moving;
    FloatImageType *fixed;
    char *command_string;
    int status;
};
#else

/* External users of API */
# ifdef plastimatch1_EXPORTS
#  define plastimatch1_EXPORT __declspec(dllexport)
# else
#  define plastimatch1_EXPORT __declspec(dllimport)
# endif

typedef (void*) Plm_registration_context;
#endif

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
Plm_registration_context plm_registration_context_create ();
plastimatch1_EXPORT
void plm_registration_set_fixed (Plm_registration_context, void*);
plastimatch1_EXPORT
void plm_registration_set_moving (Plm_registration_context, void*);
plastimatch1_EXPORT
void plm_registration_set_command_string (Plm_registration_context, void*);
plastimatch1_EXPORT
void plm_registration_execute (Plm_registration_context);
plastimatch1_EXPORT
void plm_registration_get_status (Plm_registration_context);
plastimatch1_EXPORT
void plm_registration_get_warped_image (Plm_registration_context, void*);
plastimatch1_EXPORT
void plm_registration_get_vector_field (Plm_registration_context, void*);
plastimatch1_EXPORT
void plm_registration_context_destroy (Plm_registration_context);

#if defined __cplusplus
}
#endif

#endif
