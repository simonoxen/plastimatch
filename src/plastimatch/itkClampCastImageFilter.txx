/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef __itkClampCastImageFilter_txx
#define __itkClampCastImageFilter_txx

#include "itkClampCastImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkNumericTraits.h"
#include "itkObjectFactory.h"
#include "itkProgressReporter.h"

namespace itk
{

template <class TInputImage, class TOutputImage>
ClampCastImageFilter<TInputImage, TOutputImage>
::ClampCastImageFilter()
{
    /* Do nothing */
}

template <class TInputImage, class TOutputImage>
void 
ClampCastImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
    /* Do nothing */
}

template <class TInputImage, class TOutputImage>
void 
ClampCastImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData (
    const OutputImageRegionType& outputRegionForThread,
    int threadId)
{
    itkDebugMacro(<<"Actually executing");

    // Get the input and output pointers
    InputImagePointer  inputPtr  = this->GetInput();
    OutputImagePointer outputPtr = this->GetOutput(0);

    /* Get iterators */
    typedef ImageRegionConstIterator<TInputImage> InputIterator;
    typedef ImageRegionIterator<TOutputImage> OutputIterator;
    InputIterator  inIt(inputPtr, outputRegionForThread);
    OutputIterator outIt(outputPtr, outputRegionForThread);

    /* Get min and max values of output pixel type */
    OutputImagePixelType min_value = NumericTraits<
	OutputImagePixelType>::NonpositiveMin();
    OutputImagePixelType max_value = NumericTraits<
	OutputImagePixelType>::max();
          
    /* support progress methods/callbacks */
    ProgressReporter progress (this, threadId, 
	outputRegionForThread.GetNumberOfPixels());

    /* walk through image, clamping and casting each pixel */
    while (!outIt.IsAtEnd())
    {
	const InputImagePixelType value = inIt.Get();

	if (value > max_value)
	{
	    outIt.Set (max_value);
	}
	else if (value < min_value) {
	    outIt.Set (min_value);
	}
        else {
	    outIt.Set (static_cast<OutputImagePixelType> (inIt.Get()));
	}
#if defined (commentout)
	    /* GCS - why do I think I need static_cast ?? */
	if (clamp
	    && value > static_cast<InputImagePixelType>(max_value))
	{
	    outIt.Set (max_value);
	} else if (clamp 
	    && value < static_cast<InputImagePixelType>(min_value))
	{
	    outIt.Set (min_value);
	} else {
	    outIt.Set (static_cast<OutputImagePixelType> (inIt.Get()));
	}
#endif
	++inIt;
	++outIt;
	progress.CompletedPixel();
    }
}

} // end namespace itk

#endif
