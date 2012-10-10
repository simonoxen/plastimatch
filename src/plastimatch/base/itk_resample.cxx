/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImage.h"
#include "itkResampleImageFilter.h"
#include "itkAffineTransform.h"
#include "itkVectorResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"

#include "itk_image_type.h"
#include "itk_resample.h"
#include "plm_image_header.h"

template <class T>
T
vector_resample_image (T& vf_image, Plm_image_header* pih)
{
    typedef typename T::ObjectType VFImageType;
    typedef itk::VectorResampleImageFilter < VFImageType, VFImageType > FilterType;
    typedef itk::VectorLinearInterpolateImageFunction< 
            VFImageType, double >  InterpolatorType;

    typename FilterType::Pointer filter = FilterType::New();

    filter->SetOutputOrigin (pih->m_origin);
    filter->SetOutputSpacing (pih->m_spacing);
    filter->SetSize (pih->m_region.GetSize());
    filter->SetOutputDirection (pih->m_direction);

    typedef itk::AffineTransform< double, 3 > TransformType;
    TransformType::Pointer transform = TransformType::New();
    filter->SetTransform (transform);

    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    filter->SetInterpolator (interpolator);

    FloatVector3DType v;
    v[0] = v[1] = v[2] = 0;
    filter->SetDefaultPixelValue (v);

    filter->SetInput (vf_image);
    try {
        filter->Update();
    }
    catch(itk::ExceptionObject & ex) {
        printf ("Exception running vector resample filter!\n");
        std::cout << ex << std::endl;
        getchar();
        exit(1);
    }

    T out_image = filter->GetOutput();
    return out_image;
}

template <class T>
T
vector_resample_image (T& vf_image, DoublePoint3DType origin, 
                       DoubleVector3DType spacing, SizeType size)
{
    typedef typename T::ObjectType VFImageType;
    typedef itk::VectorResampleImageFilter < VFImageType, VFImageType > FilterType;
    typedef itk::VectorLinearInterpolateImageFunction< 
            VFImageType, double >  InterpolatorType;

    typename FilterType::Pointer filter = FilterType::New();

    filter->SetOutputOrigin (origin);
    filter->SetOutputSpacing (spacing);
    filter->SetSize (size);
    filter->SetOutputDirection (vf_image->GetDirection());

    typedef itk::AffineTransform< double, 3 > TransformType;
    TransformType::Pointer transform = TransformType::New();
    filter->SetTransform (transform);

    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    filter->SetInterpolator (interpolator);

    FloatVector3DType v;
    v[0] = v[1] = v[2] = 0;
    filter->SetDefaultPixelValue (v);

    filter->SetInput (vf_image);
    try {
        filter->Update();
    }
    catch(itk::ExceptionObject & ex) {
        printf ("Exception running vector resample filter!\n");
        std::cout << ex << std::endl;
        getchar();
        exit(1);
    }

    T out_image = filter->GetOutput();
    return out_image;
}

template <class T, class U>
T
vector_resample_image (T& vf_image, U& ref_image)
{
    typedef typename T::ObjectType VFImageType;
    typedef typename U::ObjectType REFImageType;

    const typename REFImageType::PointType& new_origin = ref_image->GetOrigin();
    const typename REFImageType::SpacingType& new_spacing = ref_image->GetSpacing();
    typename REFImageType::SizeType new_size = ref_image->GetLargestPossibleRegion().GetSize();

    return vector_resample_image (vf_image, new_origin, new_spacing, new_size);
}

template <class T>
T
vector_resample_image (T& image, float x_spacing,
                        float y_spacing, float z_spacing)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::VectorResampleImageFilter < ImageType, ImageType > FilterType;
    typedef itk::VectorLinearInterpolateImageFunction< 
            ImageType, double >  InterpolatorType;

    typename FilterType::Pointer filter = FilterType::New();

    const typename ImageType::SpacingType& old_spacing = image->GetSpacing();
    const typename ImageType::PointType& old_origin = image->GetOrigin();
    typename ImageType::SizeType old_size = image->GetLargestPossibleRegion().GetSize();

    typename ImageType::SpacingType spacing;
    spacing[ 0 ] = x_spacing;
    spacing[ 1 ] = y_spacing;
    spacing[ 2 ] = z_spacing;
    printf ("New spacing at %f %f %f\n", x_spacing, y_spacing, z_spacing);
    filter->SetOutputSpacing (spacing);

    float old_coverage[3];
    old_coverage[0] = (old_size[0]+1) * old_spacing[0];
    old_coverage[1] = (old_size[1]+1) * old_spacing[1];
    old_coverage[2] = (old_size[2]+1) * old_spacing[2];

    typename ImageType::SizeType size;
    size[0] = (unsigned long) (old_coverage[0] / spacing[0]);
    size[1] = (unsigned long) (old_coverage[1] / spacing[1]);
    size[2] = (unsigned long) (old_coverage[2] / spacing[2]);
    printf ("Size was %ld %ld %ld\n", old_size[0], old_size[1], old_size[2]);
    printf ("Size will be %ld %ld %ld\n", size[0], size[1], size[2]);
    filter->SetSize (size);

    float coverage[3];
    coverage[0] = (size[0]+1) * spacing[0];
    coverage[1] = (size[1]+1) * spacing[1];
    coverage[2] = (size[2]+1) * spacing[2];
    printf ("Coverage was %g %g %g\n", old_coverage[0], old_coverage[1], old_coverage[2]);
    printf ("Coverage will be %g %g %g\n", coverage[0], coverage[1], coverage[2]);

    typename ImageType::PointType origin;
    origin[0] =  old_origin[0] 
                    - (old_spacing[0]/2.0) 
                    + (spacing[0]/2.0)
                    + ((old_coverage[0]-coverage[0])/2.0);
    filter->SetOutputOrigin (origin);

    filter->SetOutputDirection (image->GetDirection());

    typedef itk::AffineTransform< double, 3 > TransformType;
    TransformType::Pointer transform = TransformType::New();
    filter->SetTransform (transform);

    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    filter->SetInterpolator (interpolator);
    FloatVector3DType v;
    v[0] = v[1] = v[2] = 0;
    filter->SetDefaultPixelValue (v);
    //filter->SetDefaultPixelValue (VectorType::Zero);

    filter->SetInput (image);
    try {
        filter->Update();
    }
    catch(itk::ExceptionObject & ex) {
        printf ("Exception running vector resample filter!\n");
        std::cout << ex << std::endl;
        getchar();
        exit(1);
    }

    T out_image = filter->GetOutput();
    return out_image;
}

template <class T>
static 
T
resample_image (
    T& image, 
    DoublePoint3DType origin, 
    DoubleVector3DType spacing, 
    SizeType size, 
    const DirectionType& direction, 
    float default_val, 
    int interp_lin)
{
    typedef typename T::ObjectType ImageType;
    typedef typename T::ObjectType::PixelType PixelType;
    typedef itk::ResampleImageFilter < ImageType, ImageType > FilterType;
    typedef itk::LinearInterpolateImageFunction< 
        ImageType, double >  LinInterpType;
    typedef itk::NearestNeighborInterpolateImageFunction < ImageType, double >  NNInterpType;

    typename FilterType::Pointer filter = FilterType::New();

    filter->SetOutputOrigin (origin);
    filter->SetOutputSpacing (spacing);
    filter->SetSize (size);
    filter->SetOutputDirection (direction);

    typedef itk::AffineTransform< double, 3 > TransformType;
    TransformType::Pointer transform = TransformType::New();
    filter->SetTransform (transform);

    typename LinInterpType::Pointer l_interpolator = LinInterpType::New();
    typename NNInterpType::Pointer nn_interpolator = NNInterpType::New();

    if (interp_lin) {
        filter->SetInterpolator (l_interpolator);
    } else {
        filter->SetInterpolator (nn_interpolator);
    }

    filter->SetDefaultPixelValue ((PixelType) default_val);

    filter->SetInput (image);
    try {
        filter->Update();
    }
    catch(itk::ExceptionObject & ex) {
        printf ("Exception running image resample filter!\n");
        std::cout << ex << std::endl;
        getchar();
        exit(1);
    }

    T out_image = filter->GetOutput();
    return out_image;
}

template <class T>
T
resample_image (
    T& image, 
    Plm_image_header* pih, 
    float default_val, 
    int interp_lin)
{
    return resample_image (image, pih->m_origin, pih->m_spacing,
        pih->m_region.GetSize(), pih->m_direction,
        default_val, interp_lin);
}

template <class T>
T
subsample_image (T& image, int x_sampling_rate,
                int y_sampling_rate, int z_sampling_rate,
                float default_val)
{
    typedef typename T::ObjectType ImageType;
    typedef typename T::ObjectType::PixelType PixelType;
    typedef itk::ResampleImageFilter < ImageType, ImageType > FilterType;
    typename FilterType::Pointer filter = FilterType::New();
    typedef itk::LinearInterpolateImageFunction< ImageType, double >  InterpolatorType;
    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

    int sampling_rate[3];
    
    sampling_rate[0] = x_sampling_rate;
    sampling_rate[1] = y_sampling_rate;
    sampling_rate[2] = z_sampling_rate;

    /* GCS TODO: Separate out default pixel value stuff */
    filter->SetInterpolator (interpolator);
    filter->SetDefaultPixelValue ((PixelType) default_val);

    const typename ImageType::SpacingType& spacing1 = image->GetSpacing();
    const typename ImageType::PointType& origin1 = image->GetOrigin();
    const typename ImageType::SizeType size1 = image->GetLargestPossibleRegion().GetSize();

    typename ImageType::SpacingType spacing;
    typename ImageType::SizeType size;
    typename ImageType::PointType origin;
    for (int i = 0; i < 3; i++) {
        spacing[i] = spacing1[i] * sampling_rate[i];
        origin[i] = origin1[i] + 0.5 * (sampling_rate[i]-1) * spacing1[i];
        size[i] = (int) ceil(((float) size1[i] / sampling_rate[i]) - 0.5);
    }

    //compute_origin_and_size (origin, size, spacing, origin1, spacing1, size1);

    filter->SetOutputOrigin (origin);
    filter->SetOutputSpacing (spacing);
    filter->SetSize (size);

    // GCS FIX: Assume direction cosines orthogonal
    filter->SetOutputDirection (image->GetDirection());

    typedef itk::AffineTransform< double, 3 > TransformType;
    TransformType::Pointer transform = TransformType::New();
    filter->SetTransform( transform );
    filter->SetInput( image ); 
    try {
        filter->Update();
    }
    catch(itk::ExceptionObject & ex) {
        printf ("Exception running image subsample filter!\n");
        std::cout << ex << std::endl;
        getchar();
        exit(1);
    }

    T out_image = filter->GetOutput();
    return out_image;
}

/* Explicit instantiations */
template PLMBASE_API DeformationFieldType::Pointer vector_resample_image (DeformationFieldType::Pointer&, Plm_image_header*);
template DeformationFieldType::Pointer vector_resample_image (DeformationFieldType::Pointer&, FloatImageType::Pointer&);
template DeformationFieldType::Pointer vector_resample_image (DeformationFieldType::Pointer&, float, float, float);

template PLMBASE_API UCharImageType::Pointer resample_image (UCharImageType::Pointer&, Plm_image_header*, float default_val, int interp_lin);
template PLMBASE_API ShortImageType::Pointer resample_image (ShortImageType::Pointer&, Plm_image_header*, float default_val, int interp_lin);
template PLMBASE_API UShortImageType::Pointer resample_image (UShortImageType::Pointer&, Plm_image_header*, float default_val, int interp_lin);
template PLMBASE_API Int32ImageType::Pointer resample_image (Int32ImageType::Pointer&, Plm_image_header*, float default_val, int interp_lin);
template PLMBASE_API UInt32ImageType::Pointer resample_image (UInt32ImageType::Pointer&, Plm_image_header*, float default_val, int interp_lin);
template PLMBASE_API FloatImageType::Pointer resample_image (FloatImageType::Pointer&, Plm_image_header*, float default_val, int interp_lin);

template PLMBASE_API UCharImageType::Pointer subsample_image (UCharImageType::Pointer&, int, int, int, float);
template PLMBASE_API ShortImageType::Pointer subsample_image (ShortImageType::Pointer&, int, int, int, float);
template PLMBASE_API UShortImageType::Pointer subsample_image (UShortImageType::Pointer&, int, int, int, float);
template PLMBASE_API Int32ImageType::Pointer subsample_image (Int32ImageType::Pointer&, int, int, int, float);
template PLMBASE_API UInt32ImageType::Pointer subsample_image (UInt32ImageType::Pointer&, int, int, int, float);
template PLMBASE_API FloatImageType::Pointer subsample_image (FloatImageType::Pointer&, int, int, int, float);
