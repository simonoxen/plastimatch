/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plmsys_h_
#define _plmsys_h_

#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "plm_int.h"
#include "plm_path.h"

typedef struct dir_list Dir_list;

/* dir_list.cxx */
API Dir_list* dir_list_create (void);
API void dir_list_destroy (Dir_list *dir_list);
API void dir_list_init (Dir_list* dl);
API Dir_list * dir_list_load (Dir_list *dir_list, const char* dir);

/* logfile.cxx */
API void logfile_open (char* log_fn);
API void logfile_close (void);
API void logfile_printf (const char* fmt, ...);
#define lprintf logfile_printf

/* plm_endian.cxx */
API void endian2_big_to_native (void* buf, unsigned long len);
API void endian2_native_to_big (void* buf, unsigned long len);
API void endian2_little_to_native (void* buf, unsigned long len);
API void endian2_native_to_little (void* buf, unsigned long len);
API void endian4_big_to_native (void* buf, unsigned long len);
API void endian4_native_to_big (void* buf, unsigned long len);
API void endian4_little_to_native (void* buf, unsigned long len);
API void endian4_native_to_little (void* buf, unsigned long len);

/* plm_fwrite.cxx */
API void plm_fwrite (
        void* buf,
        size_t size,
        size_t count,
        FILE* fp, 
        bool force_little_endian
);
#endif
