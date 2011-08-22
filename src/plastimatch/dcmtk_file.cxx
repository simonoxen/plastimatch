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
    m_pih.print ();
}

void
Dcmtk_file::load (const char *fn) {
    /* Save a copy of the filename */
    m_fn = fn;

    /* Open the file */
    OFCondition cond = m_dfile->loadFile (fn, EXS_Unknown, EGL_noChange);
    if (cond.bad()) {
	print_and_exit ("Sorry, couldn't open file as dicom: %s\n", fn);
    }

    /* Load image header */
    DcmDataset *dset = m_dfile->getDataset();    
    OFCondition ofrc;
    const char *c;
    uint16_t rows;
    uint16_t cols;

    /* ImagePositionPatient */
    ofrc = dset->findAndGetString (DCM_ImagePositionPatient, c);
    if (ofrc.good() && c) {
	float origin[3];
	int rc = parse_dicom_float3 (origin, c);
	if (!rc) {
	    this->m_pih.set_origin (origin);
	}
    }

    /* Rows, Columns */
    ofrc = dset->findAndGetUint16 (DCM_Rows, rows);
    if (ofrc.good()) {
	ofrc = dset->findAndGetUint16 (DCM_Columns, cols);
	if (ofrc.good()) {
	    int dim[3];
	    dim[0] = cols;
	    dim[1] = rows;
	    dim[2] = 1;
	    this->m_pih.set_dim (dim);
	}
    }

    /* ImageOrientationPatient */
    ofrc = dset->findAndGetString (DCM_ImageOrientationPatient, c);
    if (ofrc.good() && c) {
	float direction_cosines[9];
	int rc = parse_dicom_float6 (direction_cosines, c);
	if (!rc) {
	    /* GCS FIX: Need to find 3rd component */
	    this->m_pih.set_direction_cosines (direction_cosines);
	}
    }

    /* PixelSpacing */
    ofrc = dset->findAndGetString (DCM_PixelSpacing, c);
    if (ofrc.good() && c) {
	float spacing[3];
	int rc = parse_dicom_float2 (spacing, c);
	if (!rc) {
	    spacing[2] = 0.0;
	    this->m_pih.set_spacing (spacing);
	}
    }
}

const char*
Dcmtk_file::get_cstr (const DcmTagKey& tag_key)
{
    const char *c = 0;
    DcmDataset *dset = m_dfile->getDataset();
    if (dset->findAndGetString(tag_key, c).good() && c) {
	return c;
    }
    return 0;
}

void
dcmtk_file_test (const char *fn)
{
    Dcmtk_file df(fn);
}
