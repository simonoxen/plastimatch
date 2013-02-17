/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bspline_xform.h"
#include "plm_int.h"
#include "volume_header.h"

class Volume_header_private {
public:
};

Volume_header::Volume_header ()
{
    for (int d = 0; d < 3; d++) {
        m_dim[d] = 0;
        m_origin[d] = 0.;
        m_spacing[d] = 0.;
    }
    this->set_direction_cosines_identity ();

    this->d_ptr = new Volume_header_private;
}

Volume_header::Volume_header (
    plm_long dim[3], float origin[3], float spacing[3])
{
    this->set_dim (dim);
    this->set_origin (origin);
    this->set_spacing (spacing);
    this->set_direction_cosines_identity ();
}

Volume_header::Volume_header (
    plm_long dim[3], float origin[3], float spacing[3],
    float direction_cosines[9])
{
    this->set (dim, origin, spacing, direction_cosines);
}

Volume_header::~Volume_header ()
{
    delete this->d_ptr;
}

void
Volume_header::set_dim (const plm_long dim[3])
{
    for (unsigned int d = 0; d < 3; d++) {
	m_dim[d] = dim[d];
    }
}

plm_long*
Volume_header::get_dim ()
{
    return m_dim;
}

const plm_long*
Volume_header::get_dim () const
{
    return m_dim;
}

void
Volume_header::set_origin (const float origin[3])
{
    for (unsigned int d = 0; d < 3; d++) {
	this->m_origin[d] = origin[d];
    }
}

float*
Volume_header::get_origin ()
{
    return m_origin;
}

const float*
Volume_header::get_origin () const
{
    return m_origin;
}

void
Volume_header::set_spacing (const float spacing[3])
{
    for (unsigned int d = 0; d < 3; d++) {
	this->m_spacing[d] = spacing[d];
    }
}

float*
Volume_header::get_spacing ()
{
    return m_spacing;
}

const float*
Volume_header::get_spacing () const
{
    return m_spacing;
}

void
Volume_header::set_direction_cosines (const float direction_cosines[9])
{
    this->m_direction_cosines.set (direction_cosines);
}

void
Volume_header::set_direction_cosines (const Direction_cosines& dc)
{
    this->m_direction_cosines.set (dc);
}

void
Volume_header::set_direction_cosines_identity ()
{
    this->m_direction_cosines.set_identity ();
}

Direction_cosines&
Volume_header::get_direction_cosines ()
{
    return this->m_direction_cosines;
}

const Direction_cosines&
Volume_header::get_direction_cosines () const
{
    return this->m_direction_cosines;
}

void
Volume_header::set (
    const plm_long dim[3],
    const float origin[3],
    const float spacing[3],
    const float direction_cosines[9])
{
    this->set_dim (dim);
    this->set_origin (origin);
    this->set_spacing (spacing);
    this->set_direction_cosines (direction_cosines);
}

void
Volume_header::set (
    const plm_long dim[3],
    const float origin[3],
    const float spacing[3],
    const Direction_cosines& dc)
{
    this->set_dim (dim);
    this->set_origin (origin);
    this->set_spacing (spacing);
    this->set_direction_cosines (dc);
}

void
Volume_header::set_from_bxf (Bspline_xform *bxf)
{
    this->set (
	bxf->img_dim,
	bxf->img_origin,
	bxf->img_spacing,
	0);
}

void 
Volume_header::clone (Volume_header *dest, Volume_header *src)
{
    dest->set (src->m_dim, src->m_origin, src->m_spacing, 
        src->m_direction_cosines);
}

void 
Volume_header::clone (const Volume_header *src)
{
    this->set (src->m_dim, src->m_origin, src->m_spacing, 
        src->m_direction_cosines);
}

void 
Volume_header::get_image_center (float center[3])
{
    int d;
    for (d = 0; d < 3; d++) {
	center[d] = this->m_origin[d] 
	    + this->m_spacing[d] * (this->m_dim[d] - 1) / 2;
    }
}


void
Volume_header::print (void) const
{
    printf ("Dim =");
    for (unsigned int d = 0; d < 3; d++) {
	printf (" %ld", (long) m_dim[d]);
    }
    printf ("\nOrigin =");
    for (unsigned int d = 0; d < 3; d++) {
	printf (" %g", m_origin[d]);
    }
    printf ("\nSpacing =");
    for (unsigned int d = 0; d < 3; d++) {
	printf (" %g", m_spacing[d]);
    }
    printf ("\nDirection =");
    for (unsigned int d1 = 0; d1 < 3; d1++) {
	for (unsigned int d2 = 0; d2 < 3; d2++) {
	    printf (" %g", m_direction_cosines[d1*3+d2]);
	}
    }
    printf ("\n");
}

/* Return 1 if the two headers are the same */
int
Volume_header::compare (Volume_header *pli1, Volume_header *pli2)
{
    int d;
    for (d = 0; d < 3; d++) {
	if (pli1->m_dim[d] != pli2->m_dim[d]) return 0;
	if (pli1->m_origin[d] != pli2->m_origin[d]) return 0;
	if (pli1->m_spacing[d] != pli2->m_spacing[d]) return 0;
    }
    for (d = 0; d < 9; d++) {
	if (pli1->m_direction_cosines[d] != pli2->m_direction_cosines[d]) {
	    return 0;
	}
    }
    return 1;
}
