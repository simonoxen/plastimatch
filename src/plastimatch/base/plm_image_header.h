/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_image_header_h_
#define _plm_image_header_h_

#include "plmbase_config.h"
#include "direction_cosines.h"
#include "itk_image.h"
#include "plm_image.h"

class Bspline_xform;
class Plm_image_header;
class Volume;
class Volume_header;

/*! \brief 
 * The Plm_image_header class defines the geometry of an image.  
 * It defines image origin, spacing, dimensions, and direction cosines, 
 * but does not contain image voxels.
 */
class PLMBASE_API Plm_image_header {
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
    Plm_image_header (const Plm_image& pli) {
        this->set_from_plm_image (pli);
    }
    Plm_image_header (const Plm_image::Pointer& pli) {
        this->set_from_plm_image (pli);
    }
    Plm_image_header (const Volume_header& vh) {
        this->set (vh);
    }
    Plm_image_header (const Volume& vol) {
        this->set (vol);
    }
    Plm_image_header (const Volume* vol) {
        this->set (vol);
    }
    Plm_image_header (Volume* vol) {
        this->set (vol);
    }
    template<class T> 
        Plm_image_header (T image) {
        this->set_from_itk_image (image);
    }

public:
    int Size (int d) const { return m_region.GetSize()[d]; }
    const SizeType& GetSize (void) const { return m_region.GetSize (); }
public:
    /*! \brief Return true if the two headers are the same. Tolerance can be specified via using digits (=number of digits used for rounding values) */
    static bool compare (Plm_image_header *pli1, Plm_image_header *pli2,int digits=5);
    static double plm_round(double val, int digits);

    int dim (int d) const { return m_region.GetSize()[d]; }
    float origin (int d) const { return m_origin[d]; }
    float spacing (int d) const { return m_spacing[d]; }

public:
    void set_dim (const plm_long dim[3]);
    void set_origin (const float origin[3]);
    void set_spacing (const float spacing[3]);
    void set_direction_cosines (
        const float direction_cosines[9]);
    void set_direction_cosines (
        const Direction_cosines& dc);
    void set (const Plm_image_header& src);
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
    void set_from_plm_image (const Plm_image *pli);
    void set_from_plm_image (const Plm_image& pli);
    void set_from_plm_image (const Plm_image::Pointer& pli);
    void set_from_volume_header (const Volume_header& vh);
    void set (const Volume_header& vh);
    void set (const Volume& vol);
    void set (const Volume* vol);

    /*! \brief Expand existing geometry to contain the 
      specified point.  Only origin and dimensions can change, 
      spacing and direction cosines will stay the same. */
    void expand_to_contain (const FloatPoint3DType& position);

    /*! \brief Create a new geometry that can contain both 
      the reference and compare image, with direction cosines 
      and voxel spacing of the reference image */
    void set_geometry_to_contain (
        const Plm_image_header& reference_pih,
        const Plm_image_header& compare_pih);

    template<class T> 
        void set_from_itk_image (T image) {
        m_origin = image->GetOrigin ();
        m_spacing = image->GetSpacing ();
        m_region = image->GetLargestPossibleRegion ();
        m_direction = image->GetDirection ();
    }
    static void clone (Plm_image_header *dest, const Plm_image_header *src) {
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

    const OriginType& GetOrigin () const {
        return m_origin;
    }
    const SpacingType& GetSpacing () const {
        return m_spacing;
    }
    const ImageRegionType& GetLargestPossibleRegion () const {
        return m_region;
    }
    const DirectionType& GetDirection () const {
        return m_direction;
    }

    void print (void) const;

    FloatPoint3DType get_index (const FloatPoint3DType& pos) const;
    FloatPoint3DType get_position (const float index[3]) const;
    void get_image_center (float center[3]) const;
    /*! \brief Get the physical size of the image, from first voxel center
      to last voxel center.  Size is zero if only one voxel. */
    void get_image_extent (float extent[3]) const;
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
