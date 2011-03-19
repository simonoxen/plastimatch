/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include "plm_config.h"
#include "plm_int.h"
#include "itkImage.h"
#include "itkWarpImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkWarpVectorImageFilter.h"
#include "itk_image.h"

/* Warp the image.  
    im_in:	    the image which is warped
    vf:		    the vector field.  output size = vf size.
    default_val:    the value to use for pixels outside the volume
*/
template<class T, class U>
T 
itk_warp_image (
    T im_in, 
    DeformationFieldType::Pointer vf, 
    int linear_interp,
    U default_val)
{
    typedef typename T::ObjectType TBase;
    typedef typename T::ObjectType::PixelType PixelType;
    typedef itk::WarpImageFilter < TBase, TBase, DeformationFieldType > T_FilterType;
    typedef itk::LinearInterpolateImageFunction < TBase, double >  T_LinInterpType;
    typedef itk::NearestNeighborInterpolateImageFunction < TBase, double >  T_NNInterpType;

    T im_out = TBase::New();

    typename T_FilterType::Pointer filter = T_FilterType::New();
    typename T_LinInterpType::Pointer l_interpolator = T_LinInterpType::New();
    typename T_NNInterpType::Pointer nn_interpolator = T_NNInterpType::New();

    const typename TBase::PointType& og = vf->GetOrigin();
    const typename TBase::SpacingType& sp = vf->GetSpacing();
    const typename TBase::DirectionType& di = vf->GetDirection();

    if (linear_interp) {
	filter->SetInterpolator (l_interpolator);
    } else {
	filter->SetInterpolator (nn_interpolator);
    }
    filter->SetOutputSpacing (sp);
    filter->SetOutputOrigin (og);
    filter->SetOutputDirection (di);
    filter->SetDeformationField (vf);
    filter->SetInput (im_in);

    filter->SetEdgePaddingValue ((PixelType) default_val);
    
    try {
	filter->Update();
    } catch( itk::ExceptionObject & excp ) {
	std::cerr << "Exception thrown " << std::endl;
	std::cerr << excp << std::endl;
    }

    im_out = filter->GetOutput();
    im_out->Update();
    return im_out;
}

UCharVecImageType::Pointer 
test_itk_warp_image (
    UCharVecImageType::Pointer im_in, 
    DeformationFieldType::Pointer vf, 
    int linear_interp,                   /* Ignored */
    UCharVecType& default_val
)
{
    UCharVecImageType::Pointer im_out = UCharVecImageType::New();

    typedef itk::NearestNeighborInterpolateImageFunction < 
	UCharVecImageType, double > NNInterpType;
    NNInterpType::Pointer nn_interpolator = NNInterpType::New();

#if defined (commentout)
    typedef itk::WarpVectorImageFilter < 
	itk::VectorImage < unsigned char, 3 >,
	itk::VectorImage < unsigned char, 3 >,
	itk::Image < itk::Vector < float, 3 >, 3 >
	> WarpFilterType;
    WarpFilterType::Pointer filter = WarpFilterType::New();
#endif

#if defined (commentout)
    typedef itk::WarpVectorImageFilter < 
	UCharVecImageType, UCharVecImageType, DeformationFieldType 
	> WarpFilterType;
    WarpFilterType::Pointer filter = WarpFilterType::New();

    const UCharVecImageType::PointType& og = vf->GetOrigin();
    const UCharVecImageType::SpacingType& sp = vf->GetSpacing();
    const UCharVecImageType::DirectionType& di = vf->GetDirection();

    filter->SetInterpolator (nn_interpolator);
    filter->SetOutputSpacing (sp);
    filter->SetOutputOrigin (og);
    filter->SetOutputDirection (di);
    filter->SetDeformationField (vf);

    unsigned int num_uchar = (im_in->GetVectorLength()-1) / 8;
    for (unsigned int uchar_no = 0; uchar_no < num_uchar; uchar_no++) {
	
    }

    filter->SetInput (im_in);
    filter->SetEdgePaddingValue ((PixelType) default_val);

    try {
	filter->Update();
    } catch( itk::ExceptionObject & excp ) {
	std::cerr << "Exception thrown " << std::endl;
	std::cerr << excp << std::endl;
    }

    im_out = filter->GetOutput();
    im_out->Update();
#endif
    return im_out;
}

#if defined (commentout)
void
test_itk_warp_nope ()
{
    UCharVecImageType::Pointer im = UCharVecImageType::New ();
    DeformationFieldType::Pointer vf = DeformationFieldType::New ();

    UCharVecType default_val(4);

    itk_warp_image (im, vf, 0, default_val);
}
#endif
