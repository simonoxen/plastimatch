/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_metadata_h_
#define _itk_metadata_h_

#include "plm_config.h"

/* -----------------------------------------------------------------------
   Function prototypes
   ----------------------------------------------------------------------- */
plastimatch1_EXPORT void 
itk_metadata_set (
    itk::MetaDataDictionary *dict, 
    const char *tag, 
    const char *value
);

plastimatch1_EXPORT void 
itk_metadata_print (
    itk::MetaDataDictionary *dict
);

#endif
