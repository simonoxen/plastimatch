/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_brook_h_
#define _fdk_brook_h_

typedef struct volume Volume;
typedef struct MGHCBCT_Options_struct MGHCBCT_Options;

#if defined __cplusplus
extern "C" {
#endif

void fdk_brook_c (Volume *vol, MGHCBCT_Options *options);

#if defined __cplusplus
}
#endif

#endif
