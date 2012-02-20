/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_file.h"
#include "plm_int.h"
#include "print_and_exit.h"

Dcmtk_file::Dcmtk_file () {
    init ();
}

Dcmtk_file::Dcmtk_file (const char *fn) {
    init ();
    this->load_header (fn);
}

Dcmtk_file::~Dcmtk_file () {
    delete m_dfile;
}

void
Dcmtk_file::debug () const
{
    printf (" %s\n", m_fn.c_str());
    m_vh.print ();
}

void
Dcmtk_file::init ()
{
    m_dfile = new DcmFileFormat;
    m_fn = "";
}

DcmDataset*
Dcmtk_file::get_dataset (void) const
{
    return m_dfile->getDataset();
}

/* Look up DICOM value from tag */
const char*
Dcmtk_file::get_cstr (const DcmTagKey& tag_key) const
{
    const char *c = 0;
    DcmDataset *dset = m_dfile->getDataset();
    if (dset->findAndGetString(tag_key, c).good() && c) {
	return c;
    }
    return 0;
}

bool
Dcmtk_file::get_uint8 (const DcmTagKey& tag_key, uint8_t* val) const
{
    return m_dfile->getDataset()->findAndGetUint8(tag_key, (*val)).good();
}

bool
Dcmtk_file::get_uint16 (const DcmTagKey& tag_key, uint16_t* val) const
{
    return m_dfile->getDataset()->findAndGetUint16(tag_key, (*val)).good();
}

bool
Dcmtk_file::get_int16_array (const DcmTagKey& tag_key, 
    const int16_t** val, unsigned long* count) const
{
    const Sint16* foo;
    OFCondition rc = m_dfile->getDataset()->findAndGetSint16Array(
	tag_key, foo, count, OFFalse);
    *val = foo;
    return rc.good();
}

bool
Dcmtk_file::get_uint16_array (const DcmTagKey& tag_key, 
    const uint16_t** val, unsigned long* count) const
{
    const Uint16* foo;
    OFCondition rc = m_dfile->getDataset()->findAndGetUint16Array(
	tag_key, foo, count, OFFalse);
    *val = foo;
    return rc.good();
}

bool
Dcmtk_file::get_element (const DcmTagKey& tag_key, DcmElement* val) const
{
    return m_dfile->getDataset()->findAndGetElement(tag_key, val).good();
}

bool
Dcmtk_file::get_sequence (const DcmTagKey& tag_key, 
    DcmSequenceOfItems*& seq) const
{
    return m_dfile->getDataset()->findAndGetSequence (tag_key, seq).good();
}

void
Dcmtk_file::load_header (const char *fn) {
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
	    this->m_vh.set_origin (origin);
	}
    }

    /* Rows, Columns */
    ofrc = dset->findAndGetUint16 (DCM_Rows, rows);
    if (ofrc.good()) {
	ofrc = dset->findAndGetUint16 (DCM_Columns, cols);
	if (ofrc.good()) {
	    size_t dim[3];
	    dim[0] = cols;
	    dim[1] = rows;
	    dim[2] = 1;
	    this->m_vh.set_dim (dim);
	}
    }

    /* ImageOrientationPatient */
    ofrc = dset->findAndGetString (DCM_ImageOrientationPatient, c);
    if (ofrc.good() && c) {
	float direction_cosines[9];
	int rc = parse_dicom_float6 (direction_cosines, c);
	if (!rc) {
	    direction_cosines[6] 
		= direction_cosines[1]*direction_cosines[5] 
		- direction_cosines[2]*direction_cosines[4];
	    direction_cosines[7] 
		= direction_cosines[2]*direction_cosines[3] 
		- direction_cosines[0]*direction_cosines[5];
	    direction_cosines[8] 
		= direction_cosines[0]*direction_cosines[4] 
		- direction_cosines[1]*direction_cosines[3];
	    this->m_vh.set_direction_cosines (direction_cosines);
	}
    }

    /* PixelSpacing */
    ofrc = dset->findAndGetString (DCM_PixelSpacing, c);
    if (ofrc.good() && c) {
	float spacing[3];
	int rc = parse_dicom_float2 (spacing, c);
	if (!rc) {
	    spacing[2] = 0.0;
	    this->m_vh.set_spacing (spacing);
	}
    }
}

bool
dcmtk_file_compare_z_position (const Dcmtk_file* f1, const Dcmtk_file* f2)
{
    return f1->m_vh.m_origin[2] < f2->m_vh.m_origin[2];
}

void
dcmtk_file_test (const char *fn)
{
    Dcmtk_file df(fn);
}
