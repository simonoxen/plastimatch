/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bowtie_correction_h_
#define _bowtie_correction_h_

#include "plm_config.h"
#include "fdk_opts.h"
#include "volume.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
bowtie_correction (Volume *vol, Fdk_options *options);

#if defined __cplusplus
}
#endif

#endif
