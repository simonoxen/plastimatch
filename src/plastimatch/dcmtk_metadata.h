/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_metadata_h_
#define _dcmtk_metadata_h_

#include "plm_config.h"

class DcmDataset;
class DcmTagKey;
class Metadata;

void
dcmtk_set_metadata (
    DcmDataset *dataset, 
    const Metadata *meta, 
    const DcmTagKey& tagkey, 
    const char* default_value);

#endif
