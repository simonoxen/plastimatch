/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_image_header_h_
#define _plm_image_header_h_

#include "plm_config.h"
#include "xform.h"
#include "volume.h"
#include "itk_image.h"
#include "print_and_exit.h"

class Plm_image_header;
class Plm_image;

class Plm_image_header {
public:
    OriginType m_origin;
    SpacingType m_spacing;
    ImageRegionType m_region;
    DirectionType m_direction;

public:
    Plm_image_header () {}
    Plm_image_header (
	float origin[3], float spacing[3], int dim[3])
    {
	this->set_from_gpuit (origin, spacing, dim, 0);
    }
    Plm_image_header (
	float origin[3], float spacing[3],
	int dim[3], float direction_cosines[9])
    {
	this->set_from_gpuit (origin, spacing, dim, direction_cosines);
    }
public:
    int Size (int d) const { return m_region.GetSize()[d]; }

public:
    /* Return 1 if the two headers are the same */
    static int compare (Plm_image_header *pli1, Plm_image_header *pli2);

public:
    void plastimatch1_EXPORT set_origin (float origin[3]);
    void plastimatch1_EXPORT set_spacing (float spacing[3]);
    void plastimatch1_EXPORT set_dim (int dim[3]);
    void plastimatch1_EXPORT 
    set_from_gpuit (float gpuit_origin[3],
		    float gpuit_spacing[3],
		    int gpuit_dim[3],
		    float gpuit_direction_cosines[9]);
    void plastimatch1_EXPORT 
    set_from_gpuit_bspline (Bspline_xform *bxf);
    void plastimatch1_EXPORT 
    set_from_plm_image (Plm_image *pli);
    template<class T> 
    void set_from_itk_image (T image) {
	m_origin = image->GetOrigin ();
	m_spacing = image->GetSpacing ();
	m_region = image->GetLargestPossibleRegion ();
	m_direction = image->GetDirection ();
    }
    static void clone (Plm_image_header *dest, Plm_image_header *src) {
	dest->m_origin = src->m_origin;
	dest->m_spacing = src->m_spacing;
	dest->m_region = src->m_region;
	dest->m_direction = src->m_direction;
    }

    void plastimatch1_EXPORT get_gpuit_origin (float gpuit_origin[3]);
    void plastimatch1_EXPORT get_gpuit_spacing (float gpuit_spacing[3]);
    void plastimatch1_EXPORT get_gpuit_dim (int gpuit_dim[3]);
    void plastimatch1_EXPORT get_gpuit_direction_cosines (
	float gpuit_direction_cosines[9]);

    void plastimatch1_EXPORT print (void) const;
    void plastimatch1_EXPORT get_image_center (float center[3]);
};

/* -----------------------------------------------------------------------
   Global functions
   ----------------------------------------------------------------------- */
void
gpuit_direction_from_itk (
    float gpuit_direction_cosines[9],
    DirectionType* itk_direction
);

#endif
