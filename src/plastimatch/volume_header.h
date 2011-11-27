/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_header_h_
#define _volume_header_h_

#include "plm_config.h"
#include "direction_cosines.h"

class Bspline_xform;
class Volume;

class gpuit_EXPORT Volume_header {
public:
    size_t m_dim[3];
    float m_origin[3];
    float m_spacing[3];
    Direction_cosines m_direction_cosines;

public:
    Volume_header ()
    {
	for (int d = 0; d < 3; d++) {
	    m_dim[d] = 0;
	    m_origin[d] = 0.;
	    m_spacing[d] = 0.;
	}
	this->set_direction_cosines_identity ();
    }
    Volume_header (size_t dim[3], float origin[3], float spacing[3])
    {
	this->set_dim (dim);
	this->set_origin (origin);
	this->set_spacing (spacing);
	this->set_direction_cosines_identity ();
    }
    Volume_header (size_t dim[3], float origin[3], float spacing[3],
	float direction_cosines[9])
    {
	this->set (dim, origin, spacing, direction_cosines);
    }

public:
    /* Return 1 if the two headers are the same */
    static int compare (Volume_header *pli1, Volume_header *pli2);

public:
    void set_dim (const size_t dim[3]);
    void set_origin (const float origin[3]);
    void set_spacing (const float spacing[3]);
    void set_direction_cosines (const float direction_cosines[9]);
    void set_direction_cosines (const Direction_cosines& dc);
    void set_direction_cosines_identity ();

    void set (const size_t dim[3], const float origin[3], 
	const float spacing[3], const float dc[9]);
    void set (const size_t dim[3], const float origin[3], 
	const float spacing[3], const Direction_cosines& dc);
    void set_from_bxf (Bspline_xform *bxf);

    static void clone (Volume_header *dest, Volume_header *src) {
	dest->set (src->m_dim, src->m_origin, src->m_spacing, 
	    src->m_direction_cosines);
    }

    void clone (const Volume_header *src) {
	this->set (src->m_dim, src->m_origin, src->m_spacing, 
	    src->m_direction_cosines);
    }

    void print (void) const;
    void get_image_center (float center[3]);
};

#endif
