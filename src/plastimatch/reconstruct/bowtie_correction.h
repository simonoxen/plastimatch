/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bowtie_correction_h_
#define _bowtie_correction_h_

#include "plmreconstruct_config.h"
#include "plmbase.h"

typedef struct fdk_parms Fdk_parms;

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
bowtie_correction (Volume *vol, Fdk_parms *parms);

#if defined __cplusplus
}
#endif

#endif
