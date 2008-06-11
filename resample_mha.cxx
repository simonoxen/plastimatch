/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "itkLinearInterpolateImageFunction.h"
#include "itkImage.h"
#include "itkResampleImageFilter.h"
#include "itkAffineTransform.h"
#include "itkVectorResampleImageFilter.h"

#include "itk_image.h"

template <class T>
T
vector_resample_image (T& vf_image, DoublePointType origin, 
		       DoubleVectorType spacing, SizeType size)
{
    typedef typename T::ObjectType VFImageType;
    typedef itk::VectorResampleImageFilter < VFImageType, VFImageType > FilterType;
    typedef itk::VectorLinearInterpolateImageFunction< 
	    VFImageType, double >  InterpolatorType;

    typename FilterType::Pointer filter = FilterType::New();

    filter->SetOutputOrigin (origin);
    filter->SetOutputSpacing (spacing);
    filter->SetSize (size);

    typedef itk::AffineTransform< double, 3 > TransformType;
    TransformType::Pointer transform = TransformType::New();
    filter->SetTransform (transform);

    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    filter->SetInterpolator (interpolator);

    FloatVectorType v;
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
vector_resample_image (T& vf_image, float* origin, float* spacing, int* size)
{
    DoublePointType cpp_origin;
    DoubleVectorType cpp_spacing;
    SizeType cpp_size;
    for (int i = 0; i < 3; i++) {
	cpp_origin[i] = origin[i];
	cpp_spacing[i] = spacing[i];
	cpp_size[i] = size[i];
    }

    return vector_resample_image (vf_image, cpp_origin, cpp_spacing, cpp_size);
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
    printf ("Size was %d %d %d\n", old_size[0], old_size[1], old_size[2]);
    printf ("Size will be %d %d %d\n", size[0], size[1], size[2]);
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

    typedef itk::AffineTransform< double, 3 > TransformType;
    TransformType::Pointer transform = TransformType::New();
    filter->SetTransform (transform);

    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    filter->SetInterpolator (interpolator);
    FloatVectorType v;
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
T
resample_image (T& image, DoublePointType origin, 
		DoubleVectorType spacing, SizeType size, float default_val)
{
    typedef typename T::ObjectType ImageType;
    typedef typename T::ObjectType::PixelType PixelType;
    typedef itk::ResampleImageFilter < ImageType, ImageType > FilterType;
    typedef itk::LinearInterpolateImageFunction< 
	    ImageType, double >  InterpolatorType;

    typename FilterType::Pointer filter = FilterType::New();

    filter->SetOutputOrigin (origin);
    filter->SetOutputSpacing (spacing);
    filter->SetSize (size);

    typedef itk::AffineTransform< double, 3 > TransformType;
    TransformType::Pointer transform = TransformType::New();
    filter->SetTransform (transform);

    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    filter->SetInterpolator (interpolator);

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
resample_image (T& image, float* origin, float* spacing, int* size, float default_val)
{
    DoublePointType cpp_origin;
    DoubleVectorType cpp_spacing;
    SizeType cpp_size;
    for (int i = 0; i < 3; i++) {
	cpp_origin[i] = origin[i];
	cpp_spacing[i] = spacing[i];
	cpp_size[i] = size[i];
    }

    return resample_image (image, cpp_origin, cpp_spacing, cpp_size, default_val);
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
	size[i] = (size1[i] + 1) / sampling_rate[i];
    }

    //compute_origin_and_size (origin, size, spacing, origin1, spacing1, size1);

    filter->SetOutputOrigin (origin);
    filter->SetOutputSpacing (spacing);
    filter->SetSize(size);

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
template DeformationFieldType::Pointer vector_resample_image (DeformationFieldType::Pointer&, float*, float*, int*);
template DeformationFieldType::Pointer vector_resample_image (DeformationFieldType::Pointer&, FloatImageType::Pointer&);
template DeformationFieldType::Pointer vector_resample_image (DeformationFieldType::Pointer&, float, float, float);
template UCharImageType::Pointer resample_image (UCharImageType::Pointer&, float*, float*, int*, float default_val);
template ShortImageType::Pointer resample_image (ShortImageType::Pointer&, float*, float*, int*, float default_val);
template UShortImageType::Pointer resample_image (UShortImageType::Pointer&, float*, float*, int*, float default_val);
template FloatImageType::Pointer resample_image (FloatImageType::Pointer&, float*, float*, int*, float default_val);
template UCharImageType::Pointer subsample_image (UCharImageType::Pointer&, int, int, int, float);
template ShortImageType::Pointer subsample_image (ShortImageType::Pointer&, int, int, int, float);
template UShortImageType::Pointer subsample_image (UShortImageType::Pointer&, int, int, int, float);
template FloatImageType::Pointer subsample_image (FloatImageType::Pointer&, int, int, int, float);
