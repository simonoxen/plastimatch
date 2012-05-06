/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xform_h_
#define _xform_h_

#include "plmbase_config.h"
#include "itkTranslationTransform.h"
#include "itkVersorRigid3DTransform.h"
#include "itkQuaternionRigidTransform.h"
#include "itkAffineTransform.h"
#include "itkBSplineDeformableTransform.h"
#include "itkThinPlateSplineKernelTransform.h"

#include "itk_image_type.h"

class Bspline_xform;
class Plm_image_header;
class Xform;
class Volume;
class Volume_header;

enum XFormInternalType {
    XFORM_NONE                  = 0,
    XFORM_ITK_TRANSLATION       = 1,
    XFORM_ITK_VERSOR            = 2,
    XFORM_ITK_QUATERNION        = 3,
    XFORM_ITK_AFFINE            = 4,
    XFORM_ITK_BSPLINE           = 5,
    XFORM_ITK_TPS               = 6,
    XFORM_ITK_VECTOR_FIELD      = 7,
    XFORM_GPUIT_BSPLINE         = 8,
    XFORM_GPUIT_VECTOR_FIELD    = 9
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


class API Xform {
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
    Xform ();
    Xform (Xform& xf);
    ~Xform ();

    void clear ();

    TranslationTransformType::Pointer get_trn ();
    VersorTransformType::Pointer get_vrs ();
    QuaternionTransformType::Pointer get_quat ();
    AffineTransformType::Pointer get_aff ();
    BsplineTransformType::Pointer get_itk_bsp ();
    TpsTransformType::Pointer get_itk_tps ();
    DeformationFieldType::Pointer get_itk_vf ();
    Bspline_xform* get_gpuit_bsp ();
    Volume* get_gpuit_vf ();

    void set_trn (TranslationTransformType::Pointer trn);
    void set_vrs (VersorTransformType::Pointer vrs);
    void set_quat (QuaternionTransformType::Pointer quat);
    void set_aff (AffineTransformType::Pointer aff);
    void set_itk_bsp (BsplineTransformType::Pointer bsp);
    void set_itk_tps (TpsTransformType::Pointer tps);
    void set_itk_vf (DeformationFieldType::Pointer vf);
    void set_gpuit_bsp (Bspline_xform* xgb);
    void set_gpuit_vf (Volume* vf);

    void get_volume_header (Volume_header *vh);
    void get_grid_spacing (float grid_spacing[3]);

public:
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
};


API void xform_load (Xform *xf, const char* fn);
API void xform_save (Xform *xf, const char* fn);
API void xform_itk_bsp_init_default (Xform *xf);
API void xform_itk_bsp_set_grid (Xform *xf,
    const BsplineTransformType::OriginType bsp_origin,
    const BsplineTransformType::SpacingType bsp_spacing,
    const BsplineTransformType::RegionType bsp_region,
    const BsplineTransformType::DirectionType bsp_direction);
void xform_to_trn (Xform *xf_out, Xform *xf_in, Plm_image_header* pih);
void xform_to_vrs (Xform *xf_out, Xform *xf_in, Plm_image_header* pih);
void xform_to_quat (Xform *xf_out, Xform *xf_in, Plm_image_header* pih);
void xform_to_aff (Xform *xf_out, Xform *xf_in, Plm_image_header* pih);
API DeformationFieldType::Pointer xform_gpuit_vf_to_itk_vf (
    Volume* vf,              /* Input */
    Plm_image_header* pih    /* Input, can be null */
);
API void xform_to_itk_bsp (Xform *xf_out, Xform *xf_in, Plm_image_header* pih, float* grid_spac);
API void xform_to_itk_bsp_nobulk (Xform *xf_out, Xform *xf_in, Plm_image_header* pih, float* grid_spac);
API void xform_to_itk_vf (Xform* xf_out, Xform *xf_in, Plm_image_header* pih);
API void xform_to_itk_vf (Xform* xf_out, Xform *xf_in, FloatImageType::Pointer image);
API void xform_to_gpuit_bsp (Xform* xf_out, Xform* xf_in, Plm_image_header* pih, float* grid_spac);
API void xform_to_gpuit_vf (Xform* xf_out, Xform *xf_in, Plm_image_header* pih);

#endif
