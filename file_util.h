/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _file_util_h_
#define _file_util_h_

#include "plm_config.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
int extension_is (char* fname, char* ext);
gpuit_EXPORT
int file_exists (const char *filename);
gpuit_EXPORT
int is_directory (const char *dir);
gpuit_EXPORT
void make_directory (const char *dirname);
gpuit_EXPORT
void make_directory_recursive (const char *dirname);

#if defined __cplusplus
}
#endif

#endif
