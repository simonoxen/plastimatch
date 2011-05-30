/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xform_h_
#define _xform_h_

#include "plm_config.h"
#include "itkTranslationTransform.h"
#include "itkVersorRigid3DTransform.h"
#include "itkQuaternionRigidTransform.h"
#include "itkAffineTransform.h"
#include "itkBSplineDeformableTransform.h"
#include "itkThinPlateSplineKernelTransform.h"

#include "bspline.h"
#include "itk_image.h"
#include "print_and_exit.h"
#include "volume.h"

class Volume_header;
class Xform;

enum XFormInternalType {
    XFORM_NONE			= 0,
    XFORM_ITK_TRANSLATION	= 1,
    XFORM_ITK_VERSOR		= 2,
    XFORM_ITK_QUATERNION	= 3,
    XFORM_ITK_AFFINE		= 4,
    XFORM_ITK_BSPLINE		= 5,
    XFORM_ITK_TPS		= 6,
    XFORM_ITK_VECTOR_FIELD	= 7,
    XFORM_GPUIT_BSPLINE		= 8,
    XFORM_GPUIT_VECTOR_FIELD	= 9
};


/* itk basic transforms */
typedef itk::TranslationTransform < double, 3 > TranslationTransformType;
typedef itk::VersorRigid3DTransform < double > VersorTransformType;
typedef itk::QuaternionRigidTransform < double > QuaternionTransformType;
typedef itk::AffineTransform < double, 3 > AffineTransformType;

/* itk B-spline transforms */
const unsigned int SplineDimension = 3;
const unsigned int SplineOrder = 3;
typedef itk::BSplineDeformableTransform <
    double, SplineDimension, SplineOrder > BsplineTransformType;

/* itk thin-plate transforms */
typedef itk::ThinPlateSplineKernelTransform < 
    float, 3 > FloatTpsTransformType;
typedef itk::ThinPlateSplineKernelTransform < 
    double, 3 > DoubleTpsTransformType;
typedef DoubleTpsTransformType TpsTransformType;

class plastimatch1_EXPORT Xform {
public:
    XFormInternalType m_type;
    
    /* The actual xform is one of the following. */
    TranslationTransformType::Pointer m_trn;
    VersorTransformType::Pointer m_vrs;
    AffineTransformType::Pointer m_aff;
    QuaternionTransformType::Pointer m_quat;
    DeformationFieldType::Pointer m_itk_vf;
    BsplineTransformType::Pointer m_itk_bsp;
    TpsTransformType::Pointer m_itk_tps;
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
	m_quat = xf.m_quat;
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
		bspline_xform_free ((Bspline_xform*) m_gpuit);
	    }
	    m_gpuit = 0;
	}
	m_type = XFORM_NONE;
	m_trn = 0;
	m_vrs = 0;
	m_quat = 0;
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
    QuaternionTransformType::Pointer get_quat () {
	if (m_type != XFORM_ITK_QUATERNION) {
	    print_and_exit ("Typecast error in get_quat()\n");
	}
	return m_quat;
    }
    AffineTransformType::Pointer get_aff () {
	if (m_type != XFORM_ITK_AFFINE) {
	    print_and_exit ("Typecast error in get_aff()\n");
	}
	return m_aff;
    }
    BsplineTransformType::Pointer get_itk_bsp () {
	if (m_type != XFORM_ITK_BSPLINE) {
	    print_and_exit ("Typecast error in get_itk_bsp()\n");
	}
	return m_itk_bsp;
    }
    TpsTransformType::Pointer get_itk_tps () {
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
    Bspline_xform* get_gpuit_bsp () {
	if (m_type != XFORM_GPUIT_BSPLINE) {
	    print_and_exit ("Typecast error in get_gpuit_bsp()\n");
	}
	return (Bspline_xform*) m_gpuit;
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
    void set_quat (QuaternionTransformType::Pointer quat) {
	clear ();
	m_type = XFORM_ITK_QUATERNION;
	m_quat = quat;
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
    void set_itk_tps (TpsTransformType::Pointer tps) {
	clear ();
	m_type = XFORM_ITK_TPS;
	m_itk_tps = tps;
    }
    void set_itk_vf (DeformationFieldType::Pointer vf) {
	clear ();
	m_type = XFORM_ITK_VECTOR_FIELD;
	m_itk_vf = vf;
    }
    void set_gpuit_bsp (Bspline_xform* xgb) {
	clear ();
	m_type = XFORM_GPUIT_BSPLINE;
	m_gpuit = (void*) xgb;
    }
    void set_gpuit_vf (Volume* vf) {
	clear ();
	m_type = XFORM_GPUIT_VECTOR_FIELD;
	m_gpuit = (void*) vf;
    }
  public:
    void get_volume_header (Volume_header *vh);
};


plastimatch1_EXPORT void xform_load (Xform *xf, const char* fn);
plastimatch1_EXPORT void xform_save (Xform *xf, const char* fn);
plastimatch1_EXPORT void
xform_itk_bsp_init_default (Xform *xf);
plastimatch1_EXPORT void
xform_itk_bsp_set_grid (Xform *xf,
    const BsplineTransformType::OriginType bsp_origin,
    const BsplineTransformType::SpacingType bsp_spacing,
    const BsplineTransformType::RegionType bsp_region,
    const BsplineTransformType::DirectionType bsp_direction);
void xform_to_trn (Xform *xf_out, Xform *xf_in, Plm_image_header* pih);
void xform_to_vrs (Xform *xf_out, Xform *xf_in, Plm_image_header* pih);
void xform_to_quat (Xform *xf_out, Xform *xf_in, Plm_image_header* pih);
void xform_to_aff (Xform *xf_out, Xform *xf_in, Plm_image_header* pih);
plastimatch1_EXPORT DeformationFieldType::Pointer 
xform_gpuit_vf_to_itk_vf (
    Volume* vf,              /* Input */
    Plm_image_header* pih    /* Input, can be null */
);
plastimatch1_EXPORT void xform_to_itk_bsp (Xform *xf_out, Xform *xf_in, Plm_image_header* pih, float* grid_spac);
plastimatch1_EXPORT void xform_to_itk_bsp_nobulk (Xform *xf_out, Xform *xf_in, Plm_image_header* pih, float* grid_spac);
plastimatch1_EXPORT void xform_to_itk_vf (Xform* xf_out, Xform *xf_in, Plm_image_header* pih);
plastimatch1_EXPORT void xform_to_itk_vf (Xform* xf_out, Xform *xf_in, FloatImageType::Pointer image);
plastimatch1_EXPORT void xform_to_gpuit_bsp (Xform* xf_out, Xform* xf_in, Plm_image_header* pih, float* grid_spac);
plastimatch1_EXPORT void xform_to_gpuit_vf (Xform* xf_out, Xform *xf_in, int* dim, float* offset, float* spacing);

#endif
