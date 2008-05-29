/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include "itkImage.h"
#include "itkWarpImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itk_image.h"

/* Warp the image.  
    im_in:	    the image which is warped
    im_sz:	    used to set the size and resolution
    vf:		    the vector field
    default_val:    the value to use for pixels outside the volume
*/
template<class T>
T
itk_warp_image (T im_in, T im_sz, DeformationFieldType::Pointer vf, float default_val)
{
    typedef typename T::ObjectType TBase;
    typedef typename T::ObjectType::PixelType PixelType;
    typedef itk::WarpImageFilter < TBase, TBase, DeformationFieldType > T_FilterType;
    typedef itk::LinearInterpolateImageFunction < TBase, double >  T_InterpolatorType;

    T im_out = TBase::New();

    typename T_FilterType::Pointer filter = T_FilterType::New();
    typename T_InterpolatorType::Pointer interpolator = T_InterpolatorType::New();

    const typename TBase::PointType& og = vf->GetOrigin();
    const typename TBase::SpacingType& sp = vf->GetSpacing();

    filter->SetInterpolator (interpolator);
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

/* Explicit instantiations */
template UCharImageType::Pointer itk_warp_image (UCharImageType::Pointer im_in, UCharImageType::Pointer im_sz, DeformationFieldType::Pointer vf, float default_val);
template UShortImageType::Pointer itk_warp_image (UShortImageType::Pointer im_in, UShortImageType::Pointer im_sz, DeformationFieldType::Pointer vf, float default_val);
template ShortImageType::Pointer itk_warp_image (ShortImageType::Pointer im_in, ShortImageType::Pointer im_sz, DeformationFieldType::Pointer vf, float default_val);
template FloatImageType::Pointer itk_warp_image (FloatImageType::Pointer im_in, FloatImageType::Pointer im_sz, DeformationFieldType::Pointer vf, float default_val);
