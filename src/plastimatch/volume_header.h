/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_header_h_
#define _volume_header_h_

#include "plm_config.h"
#include "xform.h"
#include "volume.h"
#include "itk_image.h"
#include "print_and_exit.h"

class plastimatch1_EXPORT Volume_header {
public:
    float m_origin[3];
    float m_spacing[3];
    int m_dim[3];
    float m_direction_cosines[9];

public:
    Volume_header () {}
    Volume_header (int dim[3], float origin[3], float spacing[3])
    {
	this->set (dim, origin, spacing, 0);
    }
    Volume_header (int dim[3], float origin[3], float spacing[3],
	float direction_cosines[9])
    {
	this->set (dim, origin, spacing, direction_cosines);
    }

public:
    /* Return 1 if the two headers are the same */
    static int compare (Volume_header *pli1, Volume_header *pli2);

public:
    void set_dim (int dim[3]);
    void set_origin (float origin[3]);
    void set_spacing (float spacing[3]);
    void set_direction_cosines (float direction_cosines[9]);

    void set (int dim[3], float origin[3], float spacing[3],
	float direction_cosines[9]);
    void set_from_bxf (Bspline_xform *bxf);

    static void clone (Volume_header *dest, Volume_header *src) {
	dest->set (src->m_dim, src->m_origin, src->m_spacing, 
	    src->m_direction_cosines);
    }

    void print (void) const;
    void get_image_center (float center[3]);
};

#endif
