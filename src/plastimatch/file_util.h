/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _file_util_h_
#define _file_util_h_

#include "plm_config.h"
#include "plm_int.h"
#include "pstring.h"

#if (_WIN32)
#define ISSLASH(c) (((c) == '/') || ((c) == '\\'))
#else
#define ISSLASH(c) ((c) == '/')
#endif

plmsys_EXPORT
int extension_is (const char* fname, const char* ext);
plmsys_EXPORT
int file_exists (const char *filename);
plmsys_EXPORT
uint64_t file_size (const char *filename);
plmsys_EXPORT
int is_directory (const char *dir);
plmsys_EXPORT
void make_directory (const char *dirname);
plmsys_EXPORT
void make_directory_recursive (const char *dirname);
plmsys_EXPORT
FILE* make_tempfile (void);
plmsys_EXPORT
void strip_extension (char* filename);
plmsys_EXPORT
char*
file_util_dirname (const char *filename);
plmsys_EXPORT
char*
file_util_parent (const char *filename);
plmsys_EXPORT
char*
plm_getcwd (char* s, int len);
plmsys_EXPORT
int
plm_get_dir_list (const char*** f_list);

#if defined __cplusplus
plmsys_EXPORT
Pstring*
file_load (const char* filename);
plmsys_EXPORT
void make_directory_recursive (const Pstring& filename);
#endif

#endif
