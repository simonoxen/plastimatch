/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include "dicom_util.h"
#include "plm_image_header.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "slice_list.h"

class Slice_list_private {
public:
    /* Set this if we have m_pih && ct slice uids */
    bool m_loaded;

    Plm_image_header m_pih;

    /* These must be sorted in order, starting with origin slice */
    std::vector<Pstring> m_ct_slice_uids;
public:
    Slice_list_private () {
        this->m_loaded = false;
    }
};

Slice_list::Slice_list ()
{
    this->d_ptr = new Slice_list_private;
}

Slice_list::~Slice_list ()
{
    delete this->d_ptr;
}

const Plm_image_header*
Slice_list::get_image_header (void) const
{
    return &d_ptr->m_pih;
}

void
Slice_list::set_image_header (const Plm_image_header& pih)
{
    d_ptr->m_pih = pih;
    d_ptr->m_ct_slice_uids.resize (pih.dim(2));
}

const char*
Slice_list::get_slice_uid (int index) const
{
    if (!d_ptr->m_loaded) {
	return "";
    }
    
    return d_ptr->m_ct_slice_uids[index];
}

void
Slice_list::set_slice_uid (int index, const char* slice_uid)
{
    if (index >= (int) d_ptr->m_ct_slice_uids.size()) {
        print_and_exit (
            "Illegal call to Slice_list::set_slice_uid.  "
            "Index %d > Size %d.\n", 
            index, d_ptr->m_ct_slice_uids.size());
    }
    d_ptr->m_ct_slice_uids[index] = Pstring (slice_uid);
}

bool
Slice_list::slice_list_complete () const
{
    return d_ptr->m_loaded;
}

void
Slice_list::set_slice_list_complete ()
{
    d_ptr->m_loaded = true;
}

int 
Slice_list::num_slices ()
{
    if (!d_ptr->m_loaded) {
	return 0;
    }

    return d_ptr->m_pih.dim (2);
}

int
Slice_list::get_slice_index (float z) const
{
    if (!d_ptr->m_loaded) {
	return -1;
    }

    /* NOTE: This algorithm doesn't work if there are duplicate slices */
    int slice_no = ROUND_INT ((z - d_ptr->m_pih.m_origin[2]) 
	/ d_ptr->m_pih.m_spacing[2]);
    if (slice_no < 0 || slice_no >= d_ptr->m_pih.Size(2)) {
	return -1;
    }
    return slice_no;
}
