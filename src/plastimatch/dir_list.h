/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dir_list_h_
#define _dir_list_h_

#include "plm_config.h"

typedef struct dir_list Dir_list;
struct dir_list
{
    int num_entries;
    char** entries;
};


#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
Dir_list*
dir_list_create (void);

gpuit_EXPORT
void
dir_list_init (Dir_list* dl);

gpuit_EXPORT
Dir_list *
dir_list_load (Dir_list *dir_list, const char* dir);

gpuit_EXPORT
void
dir_list_destroy (Dir_list *dir_list);

#if defined __cplusplus
}
#endif

#endif
