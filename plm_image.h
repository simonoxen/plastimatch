/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_image_h_
#define _plm_image_h_

#include "volume.h"
#include "itk_image.h"
#include "print_and_exit.h"

/* -----------------------------------------------------------------------
   PlmImage class
   ----------------------------------------------------------------------- */
class PlmImageHeader;
class PlmImageHeader {
public:
    OriginType m_origin;
    SpacingType m_spacing;
    ImageRegionType m_region;
    DirectionType m_direction;

public:
    int Size (int d) const { return m_region.GetSize()[d]; }

public:
    void plastimatch1_EXPORT set_from_gpuit (float gpuit_origin[3],
			 float gpuit_spacing[3],
			 int gpuit_dim[3]);
    void plastimatch1_EXPORT cvt_to_gpuit (float gpuit_origin[3],
			 float gpuit_spacing[3],
			 int gpuit_dim[3]);
    void plastimatch1_EXPORT print (void);
    template<class T> 
    void set_from_itk_image (T image) {
	m_origin = image->GetOrigin ();
	m_spacing = image->GetSpacing ();
	m_region = image->GetLargestPossibleRegion ();
	m_direction = image->GetDirection ();
    }
};

class PlmImage;
class PlmImage {

public:

    PlmImageType m_original_type;
    PlmImageType m_type;

    /* The actual image is one of the following. */
    FloatImageType::Pointer m_itk_float;
    ShortImageType::Pointer m_itk_short;
    UCharImageType::Pointer m_itk_uchar;
    UShortImageType::Pointer m_itk_ushort;
    void* m_gpuit;

private:
    /* Please don't use copy constructors.  They suck. */
    PlmImage (PlmImage& xf) {
    }
    /* Please don't use overloaded operators.  They suck. */
    PlmImage& operator= (PlmImage& xf) {
	return *this;
    }
    void convert_itk_float ();
    void convert_gpuit_float ();

public:
    PlmImage () {
	clear ();
    }
    ~PlmImage () {
	free ();
    }

    void clear () {
	m_type = PLM_IMG_TYPE_UNDEFINED;
	m_original_type = PLM_IMG_TYPE_UNDEFINED;
	m_gpuit = 0;
    }
    void free () {
	/* GCS FIX: This doesn't actually free anything. */
	m_type = PLM_IMG_TYPE_UNDEFINED;
	m_original_type = PLM_IMG_TYPE_UNDEFINED;
	m_gpuit = 0;
    }

    FloatImageType::Pointer& itk_float () {
	convert_itk_float ();
	return m_itk_float;
    }
    Volume* gpuit_float () {
	convert_gpuit_float ();
	return (Volume*) m_gpuit;
    }
};

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
PlmImage* rad_image_load (char* fname, PlmImageType type);
void itk_roi_from_gpuit (ImageRegionType* roi, int roi_offset[3], int roi_dim[3]);

#endif
