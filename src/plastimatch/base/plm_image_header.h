/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_image_header_h_
#define _plm_image_header_h_

#include "plm_config.h"
#include "direction_cosines.h"
#include "itk_image.h"
#include "volume.h"

class Bspline_xform;
class Plm_image_header;
class Plm_image;
class Volume_header;

class plastimatch1_EXPORT Plm_image_header {
  public:
    OriginType m_origin;
    SpacingType m_spacing;
    ImageRegionType m_region;
    DirectionType m_direction;

  public:
    Plm_image_header () {}
    Plm_image_header (
        plm_long dim[3], float origin[3], float spacing[3])
    {
        this->set_from_gpuit (dim, origin, spacing, 0);
    }
    Plm_image_header (
        plm_long dim[3], float origin[3], float spacing[3],
        float direction_cosines[9])
    {
        this->set_from_gpuit (dim, origin, spacing, direction_cosines);
    }
    Plm_image_header (Plm_image *pli) {
        this->set_from_plm_image (pli);
    }
  public:
    int Size (int d) const { return m_region.GetSize()[d]; }
    const SizeType& GetSize (void) const { return m_region.GetSize (); }
  public:
    /* Return 1 if the two headers are the same */
    static int compare (Plm_image_header *pli1, Plm_image_header *pli2);

  public:
    void set_dim (const plm_long dim[3]);
    void set_origin (const float origin[3]);
    void set_spacing (const float spacing[3]);
    void set_direction_cosines (
        const float direction_cosines[9]);
    void set_direction_cosines (
        const Direction_cosines& dc);
    void set (
        const plm_long dim[3],
        const float origin[3],
        const float spacing[3],
        const Direction_cosines& dc);
    void set (
        const plm_long dim[3],
        const float origin[3],
        const float spacing[3],
        const float direction_cosines[9]);
    void set_from_gpuit (
        const plm_long dim[3],
        const float origin[3],
        const float spacing[3],
        const float direction_cosines[9]);
    void set_from_gpuit_bspline (Bspline_xform *bxf);
    void set_from_plm_image (Plm_image *pli);
    void set_from_volume_header (const Volume_header& vh);
    void set (const Volume_header& vh);

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

    Volume_header get_volume_header () const;
    void get_volume_header (Volume_header *vh) const;
    void get_origin (float origin[3]) const;
    void get_spacing (float spacing[3]) const;
    void get_dim (plm_long dim[3]) const;
    void get_direction_cosines (float direction_cosines[9]) const;

    void print (void) const;
    void get_image_center (float center[3]) const;
};

/* -----------------------------------------------------------------------
   Global functions
   ----------------------------------------------------------------------- */
void
direction_cosines_from_itk (
    float direction_cosines[9],
    DirectionType* itk_direction
);

#endif
