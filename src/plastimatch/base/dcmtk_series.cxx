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
#include "dcmtk_series.h"

Dcmtk_series::Dcmtk_series ()
{
}

Dcmtk_series::~Dcmtk_series ()
{
    std::list<Dcmtk_file*>::iterator it;
    for (it = m_flist.begin(); it != m_flist.end(); ++it) {
	delete (*it);
    }
}

void
Dcmtk_series::debug (void) const
{
    std::list<Dcmtk_file*>::const_iterator it;
    for (it = m_flist.begin(); it != m_flist.end(); ++it) {
	Dcmtk_file *df = (*it);
	df->debug ();
    }
}

const char*
Dcmtk_series::get_cstr (const DcmTagKey& tag_key) const
{
    return m_flist.front()->get_cstr(tag_key);
}

std::string 
Dcmtk_series::get_string (const DcmTagKey& tag_key) const
{
    const char* c = m_flist.front()->get_cstr(tag_key);
    if (!c) c = "";
    return std::string(c);
}

bool
Dcmtk_series::get_uint16 (const DcmTagKey& tag_key, uint16_t* val) const
{
    return m_flist.front()->get_uint16 (tag_key, val);
}

std::string 
Dcmtk_series::get_modality (void) const
{
    return get_string (DCM_Modality);
}

std::string 
Dcmtk_series::get_referenced_uid (void) const
{
    bool rc;
    if (this->get_modality() != "RTSTRUCT") {
	return "";
    }

    DcmItem* rfors = 0;
    rc = m_flist.front()->get_dataset()->findAndGetSequenceItem (
	DCM_ReferencedFrameOfReferenceSequence, rfors).good();
    if (!rc) {
	return "";
    }
    printf ("Found DCM_ReferencedFrameOfReferenceSequence!\n");

    DcmItem* rss = 0;
    rc = rfors->findAndGetSequenceItem (
	DCM_RTReferencedStudySequence, rss).good();
    if (!rc) {
	return "";
    }
    printf ("Found DCM_RTReferencedStudySequence!\n");

    return "";
}

void
Dcmtk_series::insert (Dcmtk_file *df)
{
    m_flist.push_back (df);
}

void
Dcmtk_series::sort (void)
{
    m_flist.sort (dcmtk_file_compare_z_position);
}
