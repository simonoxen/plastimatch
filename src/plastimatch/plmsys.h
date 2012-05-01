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
#include "plm_timer.h"

/* Please excuse the mess
 *   This monolithic file is only temporary
 */

typedef struct dir_list Dir_list;

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

/* plm_endian.cxx */
C_API void endian2_big_to_native (void* buf, unsigned long len);
C_API void endian2_native_to_big (void* buf, unsigned long len);
C_API void endian2_little_to_native (void* buf, unsigned long len);
C_API void endian2_native_to_little (void* buf, unsigned long len);
C_API void endian4_big_to_native (void* buf, unsigned long len);
C_API void endian4_native_to_big (void* buf, unsigned long len);
C_API void endian4_little_to_native (void* buf, unsigned long len);
C_API void endian4_native_to_little (void* buf, unsigned long len);

/* plm_fwrite.cxx */
C_API void plm_fwrite (
        void* buf,
        size_t size,
        size_t count,
        FILE* fp, 
        bool force_little_endian
);

/* print_and_exit.cxx */
C_API void print_and_wait (char* prompt_fmt, ...);
C_API void print_and_exit (char* prompt_fmt, ...);
#define error_printf(fmt, ...) \
    fprintf (stderr, "\nplastimatch has encountered an issue.\n" \
             "file: %s (line:%i)\n" fmt, __FILE__, __LINE__,##__VA_ARGS__)

/* file_util.h */
C_API int extension_is (const char* fname, const char* ext);
C_API int file_exists (const char *filename);
C_API uint64_t file_size (const char *filename);
C_API int is_directory (const char *dir);
C_API void make_directory (const char *dirname);
C_API void make_directory_recursive (const char *dirname);
C_API FILE* make_tempfile (void);
C_API void strip_extension (char* filename);
C_API char* file_util_dirname (const char *filename);
C_API char* file_util_parent (const char *filename);
C_API char* plm_getcwd (char* s, int len);
C_API int plm_chdir (char* s);
C_API int plm_get_dir_list (const char*** f_list);

#endif
