/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_utils_h_
#define _fdk_utils_h_

typedef struct volume Volume;
typedef struct MGHCBCT_Options_struct MGHCBCT_Options;

#if defined __cplusplus
extern "C" {
#endif

Volume* my_create_volume (MGHCBCT_Options* options);
void convert_to_hu (Volume* vol, MGHCBCT_Options* options);

#if defined __cplusplus
}
#endif

#endif
