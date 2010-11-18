/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _file_util_h_
#define _file_util_h_

#include "plm_config.h"
#include "bstrwrap.h"
#include "plm_int.h"

#if (_WIN32)
#define ISSLASH(c) (((c) == '/') || ((c) == '\\'))
#else
#define ISSLASH(c) ((c) == '/')
#endif

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
int extension_is (const char* fname, const char* ext);
gpuit_EXPORT
int file_exists (const char *filename);
gpuit_EXPORT
uint64_t file_size (const char *filename);
gpuit_EXPORT
int is_directory (const char *dir);
gpuit_EXPORT
void make_directory (const char *dirname);
gpuit_EXPORT
void make_directory_recursive (const char *dirname);
gpuit_EXPORT
void strip_extension (char* filename);
gpuit_EXPORT
char*
file_util_dirname (const char *filename);
gpuit_EXPORT
char*
file_util_parent (const char *filename);

#if defined __cplusplus
gpuit_EXPORT
CBString*
file_load (const char* filename);
#endif

#if defined __cplusplus
}
#endif

#endif
