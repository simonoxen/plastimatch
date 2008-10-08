/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_image_h_
#define _plm_image_h_

#include "volume.h"
#include "itk_image.h"
#include "print_and_exit.h"

/* -----------------------------------------------------------------------
   RadImage class
   ----------------------------------------------------------------------- */
class PlmImageHeader;
class PlmImageHeader {
public:
    OriginType m_origin;
    SpacingType m_spacing;
    ImageRegionType m_region;
public:
    int Size (int d) const { return m_region.GetSize()[d]; }
public:
    void set_from_itk (const OriginType& itk_origin,
			 const SpacingType& itk_spacing,
			 const ImageRegionType& itk_region);
    void set_from_gpuit (float gpuit_origin[3],
			 float gpuit_spacing[3],
			 int gpuit_dim[3]);
    template<class T> 
    void set_from_itk_image (T image) {
	m_origin = image->GetOrigin();
	m_spacing = image->GetSpacing();
	m_region = image->GetLargestPossibleRegion ();
    }
};

class RadImage;
class RadImage {

public:
    enum RadImageType {
	TYPE_UNDEFINED   = 0, 
	TYPE_ITK_FLOAT   = 1, 
	TYPE_ITK_SHORT   = 2, 
	TYPE_ITK_UCHAR   = 3, 
	TYPE_ITK_USHORT  = 4, 
	TYPE_GPUIT_FLOAT = 5, 
    };


public:

    RadImageType m_type;

    /* The actual image is one of the following. */
    FloatImageType::Pointer m_itk_float;
    ShortImageType::Pointer m_itk_short;
    UCharImageType::Pointer m_itk_uchar;
    UShortImageType::Pointer m_itk_ushort;
    void* m_gpuit;

private:
    /* Please don't use copy constructors.  They suck. */
    RadImage (RadImage& xf) {
    }
    /* Please don't use overloaded operators.  They suck. */
    RadImage& operator= (RadImage& xf) {
	return *this;
    }
    void convert_itk_float ();
    void convert_gpuit_float ();

public:
    RadImage () {
	clear ();
    }
    ~RadImage () {
	free ();
    }

    void clear () {
	m_type = RadImage::TYPE_UNDEFINED;
	m_gpuit = 0;
    }
    void free () {
	m_type = RadImage::TYPE_UNDEFINED;
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
RadImage* rad_image_load (char* fname, RadImage::RadImageType type);
void itk_roi_from_gpuit (ImageRegionType* roi, int roi_offset[3], int roi_dim[3]);

#endif
