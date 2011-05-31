/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "bspline_xform.h"
#include "print_and_exit.h"
#include "volume.h"
#include "volume_header.h"

void
Volume_header::set_dim (int dim[3])
{
    for (unsigned int d = 0; d < 3; d++) {
	m_dim[d] = dim[d];
    }
}

void
Volume_header::set_origin (float origin[3])
{
    for (unsigned int d = 0; d < 3; d++) {
	this->m_origin[d] = origin[d];
    }
}

void
Volume_header::set_spacing (float spacing[3])
{
    for (unsigned int d = 0; d < 3; d++) {
	this->m_spacing[d] = spacing[d];
    }
}

void
Volume_header::set_direction_cosines (float direction_cosines[9])
{
    for (unsigned int d = 0; d < 9; d++) {
	this->m_direction_cosines[d] = direction_cosines[d];
    }
}

void
Volume_header::set_direction_cosines_identity ()
{
    this->m_direction_cosines[0] = 1.f;
    this->m_direction_cosines[1] = 0.f;
    this->m_direction_cosines[2] = 0.f;
    this->m_direction_cosines[3] = 0.f;
    this->m_direction_cosines[4] = 1.f;
    this->m_direction_cosines[5] = 0.f;
    this->m_direction_cosines[6] = 0.f;
    this->m_direction_cosines[7] = 0.f;
    this->m_direction_cosines[8] = 1.f;
}

void
Volume_header::set (
    int dim[3],
    float origin[3],
    float spacing[3],
    float direction_cosines[9])
{
    this->set_dim (dim);
    this->set_origin (origin);
    this->set_spacing (spacing);
    this->set_direction_cosines (direction_cosines);
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
Volume_header::print (void) const
{
    printf ("Dim =");
    for (unsigned int d = 0; d < 3; d++) {
	printf (" %d", m_dim[d]);
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

void 
Volume_header::get_image_center (float center[3])
{
    int d;
    for (d = 0; d < 3; d++) {
	center[d] = this->m_origin[d] 
	    + this->m_spacing[d] * (this->m_dim[d] - 1) / 2;
    }
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
