/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include "itkImage.h"
#include "itkWarpImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itk_image.h"

/* Warp the image.  
    im_in:	    the image which is warped
    vf:		    the vector field.  output size = vf size.
    default_val:    the value to use for pixels outside the volume
*/
template<class T, class U>
T 
itk_warp_image (T im_in, DeformationFieldType::Pointer vf, int linear_interp,
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

    if (linear_interp) {
	filter->SetInterpolator (l_interpolator);
    } else {
	filter->SetInterpolator (nn_interpolator);
    }
    filter->SetOutputSpacing (sp);
    filter->SetOutputOrigin (og);
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

#if defined (commentout)
UCharImageType::Pointer itk_warp_image (UCharImageType::Pointer im_in, DeformationFieldType::Pointer vf, int linear_interp, unsigned char default_val)
{
    return itk_warp_image_template (im_in, vf, linear_interp, default_val);
}
ShortImageType::Pointer itk_warp_image (ShortImageType::Pointer im_in, DeformationFieldType::Pointer vf, int linear_interp, short default_val)
{
    return itk_warp_image_template (im_in, vf, linear_interp, default_val);
}
FloatImageType::Pointer itk_warp_image (FloatImageType::Pointer im_in, DeformationFieldType::Pointer vf, int linear_interp, float default_val)
{
    return itk_warp_image_template (im_in, vf, linear_interp, default_val);
}
#endif

/* Explicit instantiations */
template plastimatch1_EXPORT UCharImageType::Pointer itk_warp_image (UCharImageType::Pointer im_in, DeformationFieldType::Pointer vf, int linear_interp, unsigned char default_val);
template plastimatch1_EXPORT ShortImageType::Pointer itk_warp_image (ShortImageType::Pointer im_in, DeformationFieldType::Pointer vf, int linear_interp, short default_val);
template plastimatch1_EXPORT FloatImageType::Pointer itk_warp_image (FloatImageType::Pointer im_in, DeformationFieldType::Pointer vf, int linear_interp, float default_val);
