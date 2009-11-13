/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _file_util_h_
#define _file_util_h_

#if defined __cplusplus
extern "C" {
#endif

int extension_is (char* fname, char* ext);
int file_exists (const char *filename);
int is_directory (char *dir);
void make_directory (const char *dirname);
void make_directory_recursive (const char *dirname);

#if defined __cplusplus
}
#endif

#endif
