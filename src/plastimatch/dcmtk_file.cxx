/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>

#define HAVE_CONFIG_H 1               /* Needed for debian */
#include "dcmtk/config/osconfig.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_file.h"

Dcmtk_file::Dcmtk_file () {
}

Dcmtk_file::Dcmtk_file (const char *fn) {
    this->load (fn);
}

Dcmtk_file::~Dcmtk_file () {
}

void
Dcmtk_file::load (const char *fn) {
    DcmFileFormat dfile;
    DcmObject *dset = &dfile;
    dset = dfile.getDataset();
}

void
dcmtk_file_test (const char *fn)
{
    Dcmtk_file df(fn);
}
