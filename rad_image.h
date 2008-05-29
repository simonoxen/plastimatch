/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rad_image_h_
#define _rad_image_h_

#include "volume.h"
#include "itk_image.h"
#include "print_and_exit.h"


class RadImage;

/* -----------------------------------------------------------------------
   RadImage class
   ----------------------------------------------------------------------- */
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


#endif
