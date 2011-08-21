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
#include "print_and_exit.h"

Dcmtk_file::Dcmtk_file () {
    init ();
}

Dcmtk_file::Dcmtk_file (const char *fn) {
    init ();
    this->load (fn);
}

Dcmtk_file::~Dcmtk_file () {
    delete m_dfile;
}

void
Dcmtk_file::init ()
{
    m_dfile = new DcmFileFormat;
    m_fn = "";
}

void
Dcmtk_file::debug () const
{
    printf (" %s\n", m_fn.c_str());
}

void
Dcmtk_file::load (const char *fn) {
    OFCondition cond = m_dfile->loadFile (fn, EXS_Unknown, EGL_noChange);
    if (cond.bad()) {
	print_and_exit ("Sorry, couldn't open file as dicom: %s\n", fn);
    }
    m_fn = fn;
}

const char*
Dcmtk_file::get_cstr (const DcmTagKey& tag_key)
{
    const char *c = 0;
    DcmDataset *m_dset = m_dfile->getDataset();
    if (m_dset->findAndGetString(tag_key, c).good() && c) {
	return c;
    }
    return 0;
}

void
dcmtk_file_test (const char *fn)
{
    Dcmtk_file df(fn);
}
