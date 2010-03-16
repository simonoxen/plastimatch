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
    int Size (int d) const { return m_region.GetSize()[d]; }

public:
    /* Return 1 if the two headers are the same */
    static int compare (Plm_image_header *pli1, Plm_image_header *pli2);

public:
    void plastimatch1_EXPORT 
    set_from_gpuit (float gpuit_origin[3],
		    float gpuit_spacing[3],
		    int gpuit_dim[3],
		    float gpuit_direction_cosines[9]);
    void plastimatch1_EXPORT 
    set_from_gpuit_bspline (BSPLINE_Xform *bxf);
    void plastimatch1_EXPORT 
    set_from_plm_image (Plm_image *pli);
    template<class T> 
    void set_from_itk_image (T image) {
	m_origin = image->GetOrigin ();
	m_spacing = image->GetSpacing ();
	m_region = image->GetLargestPossibleRegion ();
	m_direction = image->GetDirection ();
    }

    void plastimatch1_EXPORT get_gpuit_origin (float gpuit_origin[3]);
    void plastimatch1_EXPORT get_gpuit_spacing (float gpuit_spacing[3]);
    void plastimatch1_EXPORT get_gpuit_dim (int gpuit_dim[3]);
    void plastimatch1_EXPORT get_gpuit_direction_cosines (
	float gpuit_direction_cosines[9]);

    void plastimatch1_EXPORT print (void) const;
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
