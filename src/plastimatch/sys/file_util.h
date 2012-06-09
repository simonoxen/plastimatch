/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _file_util_h_
#define _file_util_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmsys.h"
 */

#include "plmsys_config.h"
#include <stdio.h>
#include "sys/plm_int.h"

PLMSYS_C_API int extension_is (const char* fname, const char* ext);
PLMSYS_C_API int file_exists (const char *filename);
PLMSYS_C_API uint64_t file_size (const char *filename);
PLMSYS_C_API int is_directory (const char *dir);
PLMSYS_C_API void make_directory (const char *dirname);
PLMSYS_C_API void make_directory_recursive (const char *dirname);
PLMSYS_C_API FILE* make_tempfile (void);
PLMSYS_C_API void strip_extension (char* filename);
PLMSYS_C_API char* file_util_dirname (const char *filename);
PLMSYS_C_API char* file_util_parent (const char *filename);
PLMSYS_C_API char* plm_getcwd (char* s, int len);
PLMSYS_C_API int plm_chdir (char* s);
PLMSYS_C_API int plm_get_dir_list (const char*** f_list);
PLMSYS_API std::string compose_filename (const char *a, const char *b);

#endif
