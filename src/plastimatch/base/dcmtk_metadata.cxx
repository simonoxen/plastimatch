/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_file.h"
#include "dcmtk_metadata.h"
#include "dcmtk_save.h"
#include "dcmtk_series.h"
#include "metadata.h"

void
dcmtk_set_metadata (
    DcmDataset *dataset, 
    const Metadata *meta, 
    const DcmTagKey& tagkey, 
    const char* default_value)
{
    const char* md = meta->get_metadata_ (
        tagkey.getGroup(), tagkey.getElement());
    if (md) {
        dataset->putAndInsertString (tagkey, md);
    } else {
        dataset->putAndInsertString (tagkey, "");
    }
}
