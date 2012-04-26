/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plmsys_h_
#define _plmsys_h_

#include "plmsys_config.h"

#include <stdlib.h>
#include <stdio.h>
#include "plm_int.h"
#include "plm_path.h"

class Plm_timer;
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

/* print_and_exit.cxx */
API void print_and_wait (char* prompt_fmt, ...);
API void print_and_exit (char* prompt_fmt, ...);
#define error_printf(fmt, ...) \
    fprintf (stderr, "\nplastimatch has encountered an issue.\n" \
             "file: %s (line:%i)\n" fmt, __FILE__, __LINE__,##__VA_ARGS__)

/* file_util.h */
API int extension_is (const char* fname, const char* ext);
API int file_exists (const char *filename);
API uint64_t file_size (const char *filename);
API int is_directory (const char *dir);
API void make_directory (const char *dirname);
API void make_directory_recursive (const char *dirname);
API FILE* make_tempfile (void);
API void strip_extension (char* filename);
API char* file_util_dirname (const char *filename);
API char* file_util_parent (const char *filename);
API char* plm_getcwd (char* s, int len);
API int plm_chdir (char* s);
API int plm_get_dir_list (const char*** f_list);

/* plm_timer.cxx */
API Plm_timer* plm_timer_create ();
API void plm_timer_destroy (Plm_timer *timer);
API double plm_timer_report (Plm_timer *timer);
API void plm_timer_start (Plm_timer *timer);


#endif
