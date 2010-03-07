/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xform_h_
#define _xform_h_

#include "itkTranslationTransform.h"
#include "itkVersorRigid3DTransform.h"
#include "itkAffineTransform.h"
#include "itkBSplineDeformableTransform.h"
#include "itkThinPlateSplineKernelTransform.h"

#include "bspline.h"
#include "volume.h"
#include "itk_image.h"
#include "plm_registration.h"
#include "print_and_exit.h"
#include "plm_config.h"

class Xform;

enum XFormInternalType {
    XFORM_NONE			= 0,
    XFORM_ITK_TRANSLATION	= 1,
    XFORM_ITK_VERSOR		= 2,
    XFORM_ITK_AFFINE		= 3,
    XFORM_ITK_BSPLINE		= 4,
    XFORM_ITK_TPS		= 5,
    XFORM_ITK_VECTOR_FIELD	= 6,
    XFORM_GPUIT_BSPLINE		= 7,
    XFORM_GPUIT_VECTOR_FIELD	= 8
};


/* itk basic transforms */
typedef itk::TranslationTransform < double, Dimension > TranslationTransformType;
typedef itk::VersorRigid3DTransform < double > VersorTransformType;
typedef itk::AffineTransform < double, Dimension > AffineTransformType;

/* itk B-spline transforms */
const unsigned int SplineDimension = Dimension;
const unsigned int SplineOrder = 3;
typedef double CoordinateRepType;
typedef itk::BSplineDeformableTransform <
		    CoordinateRepType,
		    SplineDimension,
		    SplineOrder > BsplineTransformType;

typedef itk::ThinPlateSplineKernelTransform< CoordinateRepType, Dimension> TPSTransformType;

#if defined (commentout)
typedef struct Xform_GPUIT_Bspline_struct Xform_GPUIT_Bspline;
struct Xform_GPUIT_Bspline_struct {
public:
    BSPLINE_Parms parms;
//    float img_origin[3];         /* Position of first vox in ROI (in mm) */
//    float img_spacing[3];    /* Size of voxels (in mm) */
};
#endif

class Xform {
public:
    XFormInternalType m_type;
    
    /* The actual xform is one of the following. */
    TranslationTransformType::Pointer m_trn;
    VersorTransformType::Pointer m_vrs;
    AffineTransformType::Pointer m_aff;
    DeformationFieldType::Pointer m_itk_vf;
    BsplineTransformType::Pointer m_itk_bsp;
    TPSTransformType::Pointer m_itk_tps;
    void* m_gpuit;

public:
    Xform () {
	m_gpuit = 0;
	clear ();
    }
    Xform (Xform& xf) {
	*this = xf;
    }
    ~Xform () {
	clear ();
    }
    Xform& operator= (Xform& xf) {
	m_type = xf.m_type;
	m_trn = xf.m_trn;
	m_vrs = xf.m_vrs;
	m_aff = xf.m_aff;
	m_itk_vf = xf.m_itk_vf;
	m_itk_bsp = xf.m_itk_bsp;
	m_itk_tps = xf.m_itk_tps;
	m_gpuit = xf.m_gpuit;                  /* Shallow copy */
	return *this;
    }
    void clear () {
	if (m_gpuit) {
	    if (m_type == XFORM_GPUIT_VECTOR_FIELD) {
		volume_destroy ((Volume*) m_gpuit);
	    } else {
		bspline_xform_free ((BSPLINE_Xform*) m_gpuit);
	    }
	    m_gpuit = 0;
	}
	m_type = XFORM_NONE;
	m_trn = 0;
	m_vrs = 0;
	m_aff = 0;
	m_itk_bsp = 0;
	m_itk_tps = 0;
	m_itk_vf = 0;
    }
    TranslationTransformType::Pointer get_trn () {
	if (m_type != XFORM_ITK_TRANSLATION) {
	    print_and_exit ("Typecast error in get_trn()\n");
	}
	return m_trn;
    }
    VersorTransformType::Pointer get_vrs () {
	if (m_type != XFORM_ITK_VERSOR) {
	    printf ("Got type = %d\n", m_type);
	    print_and_exit ("Typecast error in get_vrs ()\n");
	}
	return m_vrs;
    }
    AffineTransformType::Pointer get_aff () {
	if (m_type != XFORM_ITK_AFFINE) {
	    print_and_exit ("Typecast error in get_aff()\n");
	}
	return m_aff;
    }
    BsplineTransformType::Pointer get_bsp () {
	if (m_type != XFORM_ITK_BSPLINE) {
	    print_and_exit ("Typecast error in get_bsp()\n");
	}
	return m_itk_bsp;
    }
    TPSTransformType::Pointer get_itk_tps () {
	if (m_type != XFORM_ITK_TPS) {
	    print_and_exit ("Typecast error in get_tps()\n");
	}
	return m_itk_tps;
    }
    DeformationFieldType::Pointer get_itk_vf () {
	if (m_type != XFORM_ITK_VECTOR_FIELD) {
	    print_and_exit ("Typecast error in get_itk_vf()\n");
	}
	return m_itk_vf;
    }
    BSPLINE_Xform* get_gpuit_bsp () {
	if (m_type != XFORM_GPUIT_BSPLINE) {
	    print_and_exit ("Typecast error in get_gpuit_bsp()\n");
	}
	return (BSPLINE_Xform*) m_gpuit;
    }
    Volume* get_gpuit_vf () {
	if (m_type != XFORM_GPUIT_VECTOR_FIELD) {
	    print_and_exit ("Typecast error in get_gpuit_vf()\n");
	}
	return (Volume*) m_gpuit;
    }
    void set_trn (TranslationTransformType::Pointer trn) {
	clear ();
	m_type = XFORM_ITK_TRANSLATION;
	m_trn = trn;
    }
    void set_vrs (VersorTransformType::Pointer vrs) {
	clear ();
	m_type = XFORM_ITK_VERSOR;
	m_vrs = vrs;
    }
    void set_aff (AffineTransformType::Pointer aff) {
	clear ();
	m_type = XFORM_ITK_AFFINE;
	m_aff = aff;
    }
    void set_itk_bsp (BsplineTransformType::Pointer bsp) {
	/* Do not clear */
	m_type = XFORM_ITK_BSPLINE;
	m_itk_bsp = bsp;
    }
    void set_itk_tps (TPSTransformType::Pointer tps) {
	clear ();
	m_type = XFORM_ITK_TPS;
	m_itk_tps = tps;
    }
    void set_itk_vf (DeformationFieldType::Pointer vf) {
	clear ();
	m_type = XFORM_ITK_VECTOR_FIELD;
	m_itk_vf = vf;
    }
    void set_gpuit_bsp (BSPLINE_Xform* xgb) {
	clear ();
	m_type = XFORM_GPUIT_BSPLINE;
	m_gpuit = (void*) xgb;
    }
    void set_gpuit_vf (Volume* vf) {
	clear ();
	m_type = XFORM_GPUIT_VECTOR_FIELD;
	m_gpuit = (void*) vf;
    }
};

plastimatch1_EXPORT void load_xform (Xform *xf, char* fn);
plastimatch1_EXPORT void save_xform (Xform *xf, char* fn);
void xform_to_trn (Xform *xf_out, Xform *xf_in, Plm_image_header* pih);
void xform_to_vrs (Xform *xf_out, Xform *xf_in, Plm_image_header* pih);
void xform_to_aff (Xform *xf_out, Xform *xf_in, Plm_image_header* pih);
plastimatch1_EXPORT DeformationFieldType::Pointer 
xform_gpuit_vf_to_itk_vf (
    Volume* vf,            /* Input */
    Plm_image_header* pih    /* Input, can be null */
);
plastimatch1_EXPORT void xform_to_itk_bsp (Xform *xf_out, Xform *xf_in, Plm_image_header* pih, float* grid_spac);
plastimatch1_EXPORT void xform_to_itk_bsp_nobulk (Xform *xf_out, Xform *xf_in, Plm_image_header* pih, float* grid_spac);
plastimatch1_EXPORT void xform_to_itk_vf (Xform* xf_out, Xform *xf_in, Plm_image_header* pih);
plastimatch1_EXPORT void xform_to_itk_vf (Xform* xf_out, Xform *xf_in, FloatImageType::Pointer image);
plastimatch1_EXPORT void xform_to_gpuit_bsp (Xform* xf_out, Xform* xf_in, Plm_image_header* pih, float* grid_spac);
void xform_to_gpuit_vf (Xform* xf_out, Xform *xf_in, int* dim, float* offset, float* pix_spacing);
//template<class T> void xform_transform_point (T point_out, Xform* xf_in, T point_in);
plastimatch1_EXPORT void xform_transform_point (FloatPointType* point_out, Xform* xf_in, FloatPointType point_in);

#endif
