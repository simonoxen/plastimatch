/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dir_list_h_
#define _dir_list_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmsys.h"
 */

#include "plmsys_config.h"

typedef struct dir_list Dir_list;
struct dir_list
{
    int num_entries;
    char** entries;
};

/* dir_list.cxx */
C_API Dir_list* dir_list_create (void);
C_API void dir_list_destroy (Dir_list *dir_list);
C_API void dir_list_init (Dir_list* dl);
C_API Dir_list * dir_list_load (Dir_list *dir_list, const char* dir);

/* logfile.cxx */
C_API void logfile_open (char* log_fn);
C_API void logfile_close (void);
C_API void logfile_printf (const char* fmt, ...);
#define lprintf logfile_printf


#endif
