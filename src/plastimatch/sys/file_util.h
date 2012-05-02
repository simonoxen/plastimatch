/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _file_util_h_
#define _file_util_h_

#include "plmsys_config.h"
#include "sys/plm_int.h"

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
