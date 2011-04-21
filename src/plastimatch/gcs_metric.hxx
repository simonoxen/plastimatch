/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: ra_metric.txx,v $
  Language:  C++
  Date:      $Date: 2003/10/09 19:44:40 $
  Version:   $Revision: 1.45 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _ra_metric_txx
#define _ra_metric_txx

#include <time.h>
#include "itk_image.h"
#include "gcs_metric.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#define PLAY_WITH_METRIC 0

namespace itk
{

/*
 * Constructor
 */
template <class TFixedImage, class TMovingImage> 
GCSMetric<TFixedImage,TMovingImage>
::GCSMetric()
{
  itkDebugMacro("Constructor");
}

/*
 * Get the match Measure
 */
template <class TFixedImage, class TMovingImage> 
typename GCSMetric<TFixedImage,TMovingImage>::MeasureType
GCSMetric<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
    itkDebugMacro("GetValue( " << parameters << " ) ");
    FixedImageConstPointer fixedImage = this->GetFixedImage();

    if( !fixedImage ) {
	itkExceptionMacro( << "Fixed image has not been assigned" );
    }

    typedef itk::ImageRegionConstIteratorWithIndex<FixedImageType>
	    FixedIteratorType;
    FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );
    typename FixedImageType::IndexType index;
    MeasureType measure = NumericTraits< MeasureType >::Zero;

#if (PLAY_WITH_METRIC)
    FILE* fp = fopen("foo.txt","w");
#endif

    this->m_NumberOfPixelsCounted = 0;
    this->SetTransformParameters( parameters );
    while(!ti.IsAtEnd()) {
	index = ti.GetIndex();
	typename Superclass::InputPointType inputPoint;
	fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );
	if( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) ){
	    ++ti;
#if (PLAY_WITH_METRIC)
	    if (index[2] == 40)
		fprintf (fp, "%d %d %g %g\n",index[0],index[1],-1200.0,-1200.0);
#endif
	    continue;
	}

	typename Superclass::OutputPointType transformedPoint
		= this->m_Transform->TransformPoint( inputPoint );
	if( this->m_MovingImageMask 
	    && !this->m_MovingImageMask->IsInside( transformedPoint ) ) {
	    ++ti;
#if (PLAY_WITH_METRIC)
	    if (index[2] == 40)
		fprintf (fp, "%d %d %g %g\n",index[0],index[1],-1100.0,-1100.0);
#endif
	    continue;
	}

	if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) ) {
	    const RealType movingValue 
		    = this->m_Interpolator->Evaluate( transformedPoint );
	    const RealType fixedValue = ti.Get();
	    this->m_NumberOfPixelsCounted++;
	    const RealType diff = movingValue - fixedValue;
#if (PLAY_WITH_METRIC)
	    if (index[2] == 40)
		fprintf (fp, "%d %d %g %g\n",index[0],index[1],movingValue,fixedValue);
#endif
	    measure += diff * diff; 
	} else {
#if (PLAY_WITH_METRIC)
	    if (index[2] == 40)
		fprintf (fp, "%d %d %g %g\n",index[0],index[1],-1050.0,-1050.0);
#endif
	}
	++ti;
    }

    if( !this->m_NumberOfPixelsCounted ) {
	itkExceptionMacro(<<"All the points mapped to outside of the moving image");
    } else {
	measure /= this->m_NumberOfPixelsCounted;
    }
    
#if (PLAY_WITH_METRIC)
    fclose (fp);
#endif

    /* GCS */
    printf ("GET VALUE\n");
#if defined (commentout)
    std::cout << parameters
	<< " "
	<< measure
	<< std::endl;
#endif

//    this->InvokeEvent(IterationEvent());
    return measure;
}


/*
 * Get the Derivative Measure
 */
template < class TFixedImage, class TMovingImage> 
void
GCSMetric<TFixedImage,TMovingImage>
::GetDerivative( const TransformParametersType & parameters,
                 DerivativeType & derivative  ) const
{

    itkDebugMacro("GetDerivative( " << parameters << " ) ");
  
    if( !this->m_GradientImage ) {
	itkExceptionMacro(<<"The gradient image is null, maybe you forgot to call Initialize()");
    }

    FixedImageConstPointer fixedImage = this->GetFixedImage();

    if( !fixedImage ) {
	itkExceptionMacro( << "Fixed image has not been assigned" );
    }

    const unsigned int ImageDimension = FixedImageType::ImageDimension;
    typedef  itk::ImageRegionConstIteratorWithIndex<
	    FixedImageType> FixedIteratorType;
    typedef  itk::ImageRegionConstIteratorWithIndex<
	    ITK_TYPENAME Superclass::GradientImageType> GradientIteratorType;

    FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );
    typename FixedImageType::IndexType index;

    this->m_NumberOfPixelsCounted = 0;
    this->SetTransformParameters( parameters );
    const unsigned int ParametersDimension = this->GetNumberOfParameters();
    derivative = DerivativeType( ParametersDimension );
    derivative.Fill(NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero);
    ti.GoToBegin();

    while(!ti.IsAtEnd()) {
	index = ti.GetIndex();
	typename Superclass::InputPointType inputPoint;
	fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );
	if( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) ) {
	    ++ti;
	    continue;
	}

	typename Superclass::OutputPointType transformedPoint 
		= this->m_Transform->TransformPoint( inputPoint );
	if( this->m_MovingImageMask
	    && !this->m_MovingImageMask->IsInside( transformedPoint)) {
	    ++ti;
	    continue;
	}

	if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) ) {
	    const RealType movingValue
		    = this->m_Interpolator->Evaluate( transformedPoint );
	    const TransformJacobianType & jacobian
		    = this->m_Transform->GetJacobian( inputPoint ); 
	    const RealType fixedValue = ti.Value();
	    this->m_NumberOfPixelsCounted++;
	    const RealType diff = movingValue - fixedValue; 

	    // Get the gradient by NearestNeighboorInterpolation: 
	    // which is equivalent to round up the point components.
	    typedef typename Superclass::OutputPointType OutputPointType;
	    typedef typename OutputPointType::CoordRepType CoordRepType;
	    typedef ContinuousIndex<CoordRepType,
		    MovingImageType::ImageDimension>
		    MovingImageContinuousIndexType;

	    MovingImageContinuousIndexType tempIndex;
	    this->m_MovingImage->TransformPhysicalPointToContinuousIndex( transformedPoint, tempIndex );

	    typename MovingImageType::IndexType mappedIndex;
	    for (unsigned int j = 0;
		 j < MovingImageType::ImageDimension; j++) {
		mappedIndex[j] = static_cast<long>(vnl_math_rnd(tempIndex[j]));
	    }

	    const GradientPixelType gradient = 
		    this->m_GradientImage->GetPixel( mappedIndex );

	    for(unsigned int par=0; par<ParametersDimension; par++) {
		RealType sum = NumericTraits< RealType >::Zero;
		for(unsigned int dim=0; dim<ImageDimension; dim++) {
		    sum += 2.0 * diff * jacobian( dim, par ) * gradient[dim];
		}
		derivative[par] += sum;
	    }
	}
	++ti;
    }

    if( !this->m_NumberOfPixelsCounted ) {
	itkExceptionMacro(<<"All the points mapped to outside of the moving image");
    } else {
	for(unsigned int i=0; i<ParametersDimension; i++) {
	    derivative[i] /= this->m_NumberOfPixelsCounted;
	}
    }
    printf ("GET DERIVATIVE\n");
}


/*
 * Get both the match Measure and theDerivative Measure 
 */
template <class TFixedImage, class TMovingImage> 
void
GCSMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative(const TransformParametersType & parameters, 
                        MeasureType & value, DerivativeType  & derivative) const
{
    double duration;
    clock_t start, finish;
    int gcs_near = 0;
    int gcs_mid = 0;
    int gcs_far = 0;
    double displacement_max = 0.0;

    itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");
    start = clock();

    if( !this->m_GradientImage )
    {
	itkExceptionMacro(<<"The gradient image is null, maybe you forgot to call Initialize()");
    }

    FixedImageConstPointer fixedImage = this->GetFixedImage();

    if( !fixedImage ) 
    {
	itkExceptionMacro( << "Fixed image has not been assigned" );
    }

    const unsigned int ImageDimension = FixedImageType::ImageDimension;

    typedef  itk::ImageRegionConstIteratorWithIndex<
	    FixedImageType> FixedIteratorType;

    typedef  itk::ImageRegionConstIteratorWithIndex<
	    ITK_TYPENAME Superclass::GradientImageType> GradientIteratorType;


    FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );

    typename FixedImageType::IndexType index;

    MeasureType measure = NumericTraits< MeasureType >::Zero;

    this->m_NumberOfPixelsCounted = 0;

    this->SetTransformParameters( parameters );

    const unsigned int ParametersDimension = this->GetNumberOfParameters();
    derivative = DerivativeType( ParametersDimension );
    derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

    ti.GoToBegin();

    //FILE* fp = fopen ("itkgrad.txt", "w");
    while(!ti.IsAtEnd())
    {

	index = ti.GetIndex();
    
	typename Superclass::InputPointType inputPoint;
	fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

	if( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
	{
	    ++ti;
	    continue;
	}

	typename Superclass::OutputPointType transformedPoint = this->m_Transform->TransformPoint( inputPoint );

	if( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
	{
	    ++ti;
	    continue;
	}

	if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
	{
	    const RealType movingValue  = this->m_Interpolator->Evaluate( transformedPoint );

	    const TransformJacobianType & jacobian =
		    this->m_Transform->GetJacobian( inputPoint ); 

      
	    const RealType fixedValue     = ti.Value();
	    this->m_NumberOfPixelsCounted++;

	    const RealType diff = movingValue - fixedValue; 
	    measure += diff * diff;

	    /* GCS Debugging stuff */
	    double displacement = 0.0;
	    for (int r = 0; r < Dimension; r++) {
		displacement += fabs(inputPoint[r] - transformedPoint[r]);
	    }
#if defined (commentout)
	    if (fabs(diff) > 1200) {
		gcs_far++;
	    } else if (fabs(diff) > 200) {
		gcs_mid++;
	    } else {
		gcs_near++;
	    }
#endif
	    if (displacement > displacement_max) {
		displacement_max = displacement;
	    }
	    if (fabs(displacement) > 1.0) {
		gcs_far++;
	    } else if (fabs(displacement) > 0.1) {
		gcs_mid++;
	    } else {
		gcs_near++;
	    }

	    // Get the gradient by NearestNeighboorInterpolation: 
	    // which is equivalent to round up the point components.
	    typedef typename Superclass::OutputPointType OutputPointType;
	    typedef typename OutputPointType::CoordRepType CoordRepType;
	    typedef ContinuousIndex<CoordRepType,MovingImageType::ImageDimension>
		    MovingImageContinuousIndexType;

	    MovingImageContinuousIndexType tempIndex;
	    this->m_MovingImage->TransformPhysicalPointToContinuousIndex( transformedPoint, tempIndex );

	    typename MovingImageType::IndexType mappedIndex; 
	    for( unsigned int j = 0; j < MovingImageType::ImageDimension; j++ )
	    {
		mappedIndex[j] = static_cast<long>( vnl_math_rnd( tempIndex[j] ) );
	    }

	    const GradientPixelType gradient = 
		    this->m_GradientImage->GetPixel( mappedIndex );
	    for(unsigned int par=0; par<ParametersDimension; par++)
	    {
		RealType sum = NumericTraits< RealType >::Zero;
		for(unsigned int dim=0; dim<ImageDimension; dim++)
		{
		    sum += 2.0 * diff * jacobian( dim, par ) * gradient[dim];
		}
		derivative[par] += sum;
	    }
	}

	++ti;
    }

#if defined (commentout)
    FILE* fp = fopen ("itkderiv.txt", "w");
    for(unsigned int par=0; par<ParametersDimension; par++)
    {
	fprintf (fp, "%g\n", derivative[par]);
    }
    fclose (fp);
    exit (0);
#endif

    if( !this->m_NumberOfPixelsCounted )
    {
	itkExceptionMacro(<<"All the points mapped to outside of the moving image");
    }
    else
    {
	for(unsigned int i=0; i<ParametersDimension; i++)
	{
	    derivative[i] /= this->m_NumberOfPixelsCounted;
	}
	measure /= this->m_NumberOfPixelsCounted;
    }

    value = measure;
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;


    printf ("GET VALUE+DERIVATIVE: %6.3f [%6d %6d %6d] %6.3f [%6.3f secs]\n", 
	    value, gcs_near, gcs_mid, gcs_far, displacement_max, duration);

#if defined (commentout)
    printf ("GET VALUE+DERIVATIVE: %6.3f %d %6.3f\n", 
	    value, this->m_NumberOfPixelsCounted, duration);
    std::cout << parameters
	      << " "
	      << measure
	      << std::endl;
#endif
}

} // end namespace itk


#endif
