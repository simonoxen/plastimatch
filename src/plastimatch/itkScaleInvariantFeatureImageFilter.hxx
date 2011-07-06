/*=========================================================================
Program: ITK nSIFT Implemention (Template Source)
Module: $RCSfile: itkScaleInvariantFeatureImageFilter.txx,v $
Language: C++
Date: $Date: 2007/11/25 15:51:48 $
Version: $Revision: 1.0 $
Copyright (c) 2005,2006,2007 Warren Cheung
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
* The name of the Insight Consortium, nor the names of any consortium members,
nor of any contributors, may be used to endorse or promote products derived
from this software without specific prior written permission.
* Modified source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=========================================================================*/

//#define VERBOSE
//#define DEBUG
//#define DEBUG_VERBOSE

//#define GENERATE_KEYS 
//#define REORIENT

#define SIFT_FEATURE

//TODO:  Migrate the histogram reorientation code to the SIFT feature
// Make histogram feature special case of the sift feature with
// a single histogram.
//REORIENT NSIFT:  Keep a "global histogram" in addition to the specific ones.
// Or maybe do the global histogram/reorient phase first?

// Granularity of the Histogram causing cycles in reorientation accuracy?
// Or maybe we need to shift all the angles by half a bin width?

//Generate histograms
//compare gradient histograms
//

// Min DoG value for keypoints
// Fix Gaussian Scale
// Only iterate through the region up to 3 sigma from the point

//More advanced features for better matching?
//Simple quadrant system?


/* Based on example code from the ITK Toolkit
 * TODO:  Use resampler+identity transform+spacing change instead of scaling
 * arbitrary downscale between pyramid levels
 * may need to have a threshold
 * Gaussian filtration may fail if input image format is integer
 * Generate Pointset
 * Get Orientation
 * Generate Features

 * Test vs Noise (no need to for orientation)
 * vs Stretch 
 * vs Scale
 */
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkScaleInvariantFeatureImageFilter.h"
#include <cstdio>
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"
#include <itkLinearInterpolateImageFunction.h>
#include "itkDiscreteHessianGaussianImageFunction.h"
#include <itkHessianRecursiveGaussianImageFilter.h>  //compute Hessian
#include "itkRescaleIntensityImageFilter.h"


#ifndef SIFTKEY_CLASS
#define SIFTKEY_CLASS


const float PI=3.14159265;

namespace itk
{  

  template <class TFixedImageType, int VDimension> 

  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::ScaleInvariantFeatureImageFilter() 
  {
#ifdef DO_DOUBLE
    m_ImageScalesTestedNumber = 4;
#else
    m_ImageScalesTestedNumber = 3;
#endif
    m_ScalingFactor = 2.0;
    m_DifferenceOfGaussianTestsNumber = 3; //2
#ifdef DO_DOUBLE
    m_DoubleOriginalImage = true;
#else
    m_DoubleOriginalImage = false; //false: no double image ; true: double image
#endif
    m_HistogramBinsNumber = 36;      
    m_ErrorThreshold = 0.0;
    m_MaxFeatureDistanceRatio = 0.8;
	//m_SigmaAliasing=0.5;
    m_GaussianSigma = 1.5;  
    m_MinKeypointValue = 0.03; // if we assume image pixel value in the range [0,1] (otherwise 72)
    m_SIFTHalfWidth = 8;  // This MUST be a multiple of m_SIFTSubfeatureWidth
    m_SIFTSubfeatureWidth = 4;
    m_SIFTSubfeatureBins = 8;
    
	m_ThresholdPrincipalCurve = 172.3025; //1612.00; //412.1204(50); //172.3025(20); //53.24(5); //92.61(10);

    // Derived from above
    m_DifferenceOfGaussianImagesNumber = m_DifferenceOfGaussianTestsNumber+2;
    m_GaussianImagesNumber = m_DifferenceOfGaussianImagesNumber+1;
    m_IdentityTransform = IdentityTransformType::New();
  }

    
  template <class TFixedImageType, int VDimension> 
  double
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::GetGaussianScale( int j ) 
  {	
	  /*sigma{i}=k*sigma{i-1};
	  with k=2^(1/s);
	  but sigma_{total}^2 = sigma_{i}^2 + sigma_{i-1}^2;
	  so sigma_{i}^2 = sigma_{total}^2 - sigma_{i-1}^2;*/

	  double k=0;
	  
	  k = pow( 2.0, 1.0 / (double) m_DifferenceOfGaussianTestsNumber);
	  if (j==0){
			return m_GaussianSigma;}
	  else{
			double sig_prev=0;
			double sig_total=0;
			double variance_gauss=0;
	        sig_prev = pow( k, j-1 ) * m_GaussianSigma;
			sig_total = sig_prev * k;
			variance_gauss = sqrt( sig_total * sig_total - sig_prev * sig_prev );
			return variance_gauss;}
	
	/*std::cout<<std::endl;
	  std::cout<<"scala: "<<j<<" s:"<< m_DifferenceOfGaussianTestsNumber<<" sigma: "<<m_GaussianSigma<<std::endl;
	  std::cout<<"pow: "<<pow(2, (double) j / (double) m_DifferenceOfGaussianTestsNumber)<<std::endl;*/

    //return (pow(2, (double) j / (double) m_DifferenceOfGaussianTestsNumber) * m_GaussianSigma);

  }

  template <class TFixedImageType, int VDimension> 
  unsigned int
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::HistFeatureSize() 
  {
    unsigned int size = 1;
    // Have m_HistogramBinsNumber for each of the (VDimension-1) Orientation dimensions
    for (unsigned int i = 1; i < VDimension; ++i)
      {
	size *= m_HistogramBinsNumber;
      }
    return size;
  }

  template <class TFixedImageType, int VDimension> 
  typename ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>::GradientImageType::Pointer
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::GetHypersphericalCoordinates(typename GradientImageType::Pointer inputImg) 
  {
    typedef itk::ImageRegionIteratorWithIndex< GradientImageType > ImageIteratorType;
    typedef itk::ImageRegionConstIteratorWithIndex< GradientImageType > ConstImageIteratorType;
    typedef itk::ImageRegionConstIteratorWithIndex< TFixedImageType > ConstFixedImageIteratorType;
    typename GradientImageType::Pointer outputImg = GradientImageType::New();
    // Copy attributes
    outputImg->SetRegions(inputImg->GetLargestPossibleRegion());
    outputImg->CopyInformation( inputImg );
    outputImg->Allocate();

    ConstImageIteratorType inputIt(inputImg, inputImg->GetLargestPossibleRegion());
    ImageIteratorType outputIt(outputImg, inputImg->GetLargestPossibleRegion());
      
    for ( inputIt.GoToBegin(), outputIt.GoToBegin(); !inputIt.IsAtEnd();
	  ++inputIt, ++outputIt)
      {
	typename GradientImageType::PixelType x =  inputIt.Get();
	typename GradientImageType::PixelType p;

	// position 0 is the norm
	p[0] = x.GetNorm();

	// position 1 is arctan (x0 / x1)
	p[1] = atan2( x[0],x[1] );

	// Iterate over all the positions
	// position k  is arctan (x_k-1 / (x_k * cos p_k))	  
	for (unsigned int k = 2; k < x.Size(); ++k)
	  {
	    p[k] = atan2( x[k-1], x[k] * cos(p[k-1]));
	  }
	outputIt.Set(p);
      }

    return outputImg;
  }

  // Generates a vector with positions 1 to VDimension-1 filled with
  // histogram bin numbers from a single bin number

  template <class TFixedImageType, int VDimension> 
  typename ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>::GradientImageType::PixelType
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::BinToVector (unsigned int maxbin) 
  {
    // convert maxpos to orientation bin vector
    typename GradientImageType::PixelType maxp;
    for (unsigned int i = 1; i < VDimension; ++i) {
      maxp[i] = maxbin % m_HistogramBinsNumber;
      maxbin /= m_HistogramBinsNumber;
    }
    return maxp;
  }           

  template <class TFixedImageType, int VDimension> 
  unsigned int
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::SiftFeatureSize() 
  {
    unsigned int size = 1;
    // Have m_HistogramBinsNumber for each of the (VDimension-1) Orientation dimensions
    for (unsigned int i = 0; i < VDimension; ++i)
      {
	size *= (m_SIFTHalfWidth * 2 / m_SIFTSubfeatureWidth );
	if (i > 0)

	  size *= m_SIFTSubfeatureBins;
      }
    
    return size;
  }
  
  // Convert the delta iterator into index to the
  // start of the SIFT histogram
  template <class TFixedImageType, int VDimension> 
  unsigned int
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::DeltaToSiftIndex (int delta[]) 
  {
    unsigned int bin = 0;
    unsigned int binpos = 1;
#ifdef DEBUG_VERBOSE
    std::cerr << "Converting delta: ";
#endif
    for (unsigned int i = 0; i < VDimension; ++i) {
#ifdef DEBUG_VERBOSE
      std::cerr << delta[i];
#endif
      unsigned int tmp =  (delta[i] + m_SIFTHalfWidth) / m_SIFTSubfeatureWidth;
      
      bin += tmp * binpos;
      binpos *= (m_SIFTHalfWidth * 2 / m_SIFTSubfeatureWidth );
      
      
      //std::cout << "... tmp,bin,binpos( ";
	  //std::cout << tmp << " \n";
	  //std::cout << bin << " \n";
	  //std::cout << binpos<< " \n";      
    }
    
    for (unsigned int i = 1; i < VDimension; ++i)
    {
      bin *= m_SIFTSubfeatureBins;
      //std::cout << "bin_end: "<< bin << std::endl;
      }
    
    
#ifdef DEBUG_VERBOSE
    std::cerr << "\n";
#endif
    return bin;
  }


#ifdef GENERATE_KEYS
  template <class TFixedImageType, int VDimension> 
  typename ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>::FeatureType
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::GetSiftKey(typename GradientImageType::Pointer inputImg,
	       FixedImagePointer multImg,
	       IndexType pixelIndex) 
  {
#ifdef DEBUG_VERBOSE
    std::cerr << "GetSiftKey..." << std::endl; 
#endif
    FeatureType sifthistogram(this->SiftFeatureSize());
    sifthistogram.Fill(0);

    // delta iterates from  -m_SIFTHalfWidth to m_SIFTHalfWidth-1 
    // in each dimensions
    int delta[VDimension];
    for (int k = 0; k < VDimension; ++k) {
      delta[k] = -m_SIFTHalfWidth;
      //std::cout << "delta: "<< delta[k] << std::endl;
      
    }

    typename GradientImageType::SizeType regionSize = 
      inputImg->GetLargestPossibleRegion().GetSize();	
    
    while(1) {
      unsigned int siftbin = this->DeltaToSiftIndex(delta);
      //std::cout << "Siftbin: "<< siftbin << std::endl;
      
#ifdef DEBUG_VERBOSE
      std::cerr << "Siftbin:" << siftbin << std::endl; 
#endif
      
      // Get pixel index
      // Clamp to image edges
      IndexType tmpIndex;
      for (int k=0; k < VDimension; ++k) {
	if ((pixelIndex[k] + delta[k]) < 0) {
	  tmpIndex[k] = 0;
	} else {
	  tmpIndex[k] = pixelIndex[k] + delta[k];
	  if (tmpIndex[k] >= regionSize[k])
	    tmpIndex[k] = regionSize[k]-1;
	}
	//std::cout << "tmpIndex: "<< tmpIndex[k] << std::endl;
      }
      
#ifdef DEBUG_VERBOSE
      std::cerr << "Pixel:" << tmpIndex << std::endl; 
#endif
      typename GradientImageType::PixelType x = 
	inputImg->GetPixel(tmpIndex);
      
      // Get histogram bin
      // Iterate over all the positions
      unsigned int bin = 0;
      unsigned int binpos = 1;
      for (unsigned int k = 1; k < x.Size(); ++k)
	{
	  // Rescale from -PI to PI ->  0 to m_HistogramBinsNumber-1
	  float p;
	  p = (x[k] + PI)  * (float) m_SIFTSubfeatureBins / (2.0 * PI);
	  
	  
	  if (p < 0 || p >= m_SIFTSubfeatureBins) 
	    p = 0;
	  bin += (unsigned int) p * binpos;
	  
#ifdef DEBUG_VERBOSE
	  std::cout << " " << p;
#endif
	  binpos *= m_SIFTSubfeatureBins;
	  //std::cout << "p: "<< p << std::endl;
	  //std::cout << "bin: "<< bin << std::endl;
	  //std::cout << "binpos: "<< binpos << std::endl;
	  
	}
      
      bin += siftbin;
       //std::cout << "bin_aggiornato: "<< bin << std::endl;
      
      
      // Fill Sift Index bin
      if (bin > this->SiftFeatureSize()) {
	// VERY BAD
	std::cerr << bin << " > " << this->SiftFeatureSize() << " Warning -- Overload2\n";
      }
      sifthistogram[bin] += x[0] * multImg->GetPixel(tmpIndex);
      
#ifdef DEBUG_VERBOSE
      std::cerr << "Incrementing\n";
#endif	  
      // Increment delta
      bool resetdelta=false;
      for(int k=0; k <= VDimension; ++k) {
#ifdef DEBUG_VERBOSE
	std::cerr << delta[k];
#endif
	if (k == VDimension) {
#ifdef DEBUG_VERBOSE
	  std::cerr << "done\n";
#endif
	  resetdelta = true;
	  break; // done
	}
	// Don't want to go past m_SIFTHalfWidth-1
	if (++delta[k] < (int) m_SIFTHalfWidth) {
	  break;
	}
	delta[k] = -m_SIFTHalfWidth; // reset and increment the next pos
      }
      if(resetdelta) break;	  
    }
    
#ifdef DEBUG_VERBOSE
    std::cerr << "SIFT key: " << sifthistogram << "\n";
#endif
    return(sifthistogram);
  }
  

  // Takes a hyperspherical coordinate gradient and gaussian weights
  // returns a histogram 
  // Each orientation divides the 2PI angles into m_HistogramBinsNumber 
  // The value in each bin is the weighted magnitude
  template <class TFixedImageType, int VDimension> 
  typename ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>::FeatureType
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::GetHistogram(typename GradientImageType::Pointer inputImg,
		 FixedImagePointer multImg) 
  {
    
#ifdef ERROR_CHECK
    std::cerr << "GetHistogram ... ";
#endif
    
    FeatureType histogram(this->HistFeatureSize());

    histogram.Fill(0);
    
    typedef itk::ImageRegionConstIteratorWithIndex< GradientImageType > ConstImageIteratorType;
    typedef itk::ImageRegionConstIteratorWithIndex< TFixedImageType > ConstFixedImageIteratorType;
    
    ConstImageIteratorType inputIt(inputImg, inputImg->GetLargestPossibleRegion());
    ConstFixedImageIteratorType multIt( multImg, inputImg->GetLargestPossibleRegion());
      
    for ( inputIt.GoToBegin(), multIt.GoToBegin(); !inputIt.IsAtEnd();
	  ++inputIt, ++multIt)
      {
	typename GradientImageType::PixelType x =  inputIt.Get();
	typename TFixedImageType::PixelType m = multIt.Get();
	unsigned int bin = 0;
	unsigned int binpos = 1;
	typename GradientImageType::PixelType p;
	
	// position 0 is the norm
	p[0] = x[0];
	
	if (std::isnan(p[0]) || (p[0] == 0.0))
	  continue;
	
	// multiply by m
	p[0] *= m;
	
#ifdef DEBUG_VERBOSE
	std::cout << "Bin: ";
#endif	  
	// Iterate over all the positions
	for (unsigned int k = 1; k < x.Size(); ++k)
	  {
	    // Rescale from -PI to PI ->  0 to m_HistogramBinsNumber-1
	    p[k] = (x[k] + PI)  * (float) m_HistogramBinsNumber / (2.0 * PI);
	    
	    
	    if (p[k] < 0 || p[k] >= m_HistogramBinsNumber) 
	      p[k] = 0;
	    bin += (unsigned int) p[k] * binpos;
	    
#ifdef DEBUG_VERBOSE
	    std::cout << " " << p[k];
#endif
	    binpos *= m_HistogramBinsNumber;
	  }
#ifdef DEBUG_VERBOSE
	std::cout << " Value: " << p[0] << std::endl;
#endif
	if (bin > this->HistFeatureSize()) {
	  // VERY BAD
	  std::cerr << x << " -> " << p << "\n";
	  std::cerr << bin << " > " << this->HistFeatureSize() << " Warning -- Overload2\n";
	}
	histogram[bin] += p[0];
      }
#ifdef DEBUG
    // Print the Histogram
    std::cout << histogram << std::endl;
#endif
    
    // Since we are going to use this as a feature
    // Normalise
    float hmag = 0.0;
#ifdef REORIENT
    float maxmag = -1;
    unsigned int maxbin;
#endif
    for (unsigned int i = 0; i < this->HistFeatureSize(); ++i) {
      float mag = histogram[i]*histogram[i];
      hmag += mag;
#ifdef REORIENT
      if (maxmag < 0 || mag > maxmag) {
	maxmag = mag;
	maxbin = i;
      }
#endif
    }
    hmag = sqrt(hmag);
    
#ifdef REORIENT
    typename GradientImageType::PixelType maxp = this->BinToVector(maxbin);
    
    FeatureType histogram2(this->HistFeatureSize());
    histogram2.Fill(0);
#endif
    
    for (unsigned int i = 0; i < this->HistFeatureSize(); ++i) {
      histogram[i] /= hmag;
      
#ifdef REORIENT
      typename GradientImageType::PixelType bini = this->BinToVector(i);
      
      unsigned int bin = 0;
      unsigned int binpos = 1;
      for (unsigned int k = 1; k < VDimension; ++k) {
	bini[k] = ((int) (bini[k] - maxp[k] + m_HistogramBinsNumber)) % m_HistogramBinsNumber;
	if (bini[k] < 0 || bini[k] >= m_HistogramBinsNumber)
	  bini[k] = 0;
	bin += (int) bini[k] * binpos;
	binpos *= m_HistogramBinsNumber;
      }
      histogram2[bin] = histogram[i];
#endif
    }
    
    
#ifdef DEBUG
    std::cout << histogram << std::endl;
#ifdef REORIENT
    std::cout << "Max Bin: " << maxbin << " Max Mag: " << maxmag << std::endl;
    std::cout << "Reoriented: " << histogram2 << std::endl;
#endif
#endif
#ifdef ERROR_CHECK
    std::cerr << "OK\n ";
#endif
    
#ifdef REORIENT
    return histogram2;
#else
    return histogram;
#endif
  }
#endif

  template <class TFixedImageType, int VDimension> 
  void
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::CheckLocalExtrema(FixedImagePointer image, IndexType pixelIndex,
		      PixelType pixelValue,
		      bool &isMax, bool &isMin,
		      bool checkCentre)
  {
#ifdef ERROR_CHECK
    std::cerr << "CheckLocalExtrema ... ";
#endif
    int delta[VDimension];
    for (int k = 0; k < VDimension; ++k) {
      delta[k] = -1;
    }
    while(1) {
      bool isZero=true;
      if (!checkCentre) {
	for (int k=0; k < VDimension; ++k) {
	  if(delta[k] != 0) {
	    isZero = false;
	    break;
	  }
	}
      }
      
      if (checkCentre || !isZero) {
	// Check if not the centre
	IndexType tmpIndex;
	for (int k=0; k < VDimension; ++k) {
	  tmpIndex[k] = pixelIndex[k] + delta[k];
	}
	
	typename TFixedImageType::PixelType tmpValue = 
	  image->GetPixel(tmpIndex);
	  

	
#ifdef DEBUG_VERBOSE
	std::cout << "...Comparing to ( ";
	
	for (int k = 0; k < VDimension; ++k)
	{
	  std::cout << tmpIndex[k] << " ";
	  std::cout << pixelIndex[k] << " ";
	  std::cout << delta[k] << "\n";
	//std::cout << ") = " << tmpValue << "\n";
	}
#endif  
	// Treat as equality if within the error bound
	if (((tmpValue - pixelValue) <= m_ErrorThreshold) && 
	    ((tmpValue - pixelValue) >= -m_ErrorThreshold)) {
	  isMax = false;
	  isMin = false;
	} else	
	  if (tmpValue > pixelValue) {
	    isMax = false;
	  } else
	    if (tmpValue < pixelValue) {
	      isMin = false;
	    }
	if (!isMax && !isMin) break;
      }
      // Increment delta
      bool resetdelta=false;
      for(int k=0; k <= VDimension; ++k) {
	if (k == VDimension) {
	  resetdelta = true;
	  break; // done
	}
	if (delta[k] < 1) {
	  ++delta[k];
	  //std::cout << "...incremento delta ( ";
	  //std::cout << delta[k] << " \n";
	  break;
	}
	delta[k] = -1; // reset and increment the next pos
      }
      if(resetdelta) break;	  
    }  
#ifdef ERROR_CHECK
    std::cerr << "OK\n";
#endif
  }
  
#ifdef GENERATE_KEYS
  // Input:  Image, Gradient Image, Point
  // Output:  Vector of direction
  template <class TFixedImageType, int VDimension> 
  typename ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>::FeatureType
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::GetFeatures( FixedImagePointer fixedImage, 
		 typename GradientImageType::Pointer hgradImage,
		 PointType &point, float currScale) 
  {
#ifdef ERROR_CHECK
    std::cerr << "GetFeatures ... ";
#endif
    
    // Generate the Gaussian
    typedef GaussianImageSource<TFixedImageType> GaussianImageSourceType;
    typename GaussianImageSourceType::Pointer gaussImgSource;
    typename GaussianImageSourceType::ArrayType sigma;

    gaussImgSource = GaussianImageSourceType::New();
    gaussImgSource->SetNormalized(true);
    //gaussImgSource->SetNormalized(false);
    //gaussImgSource->SetScale(255);  
    gaussImgSource->SetSpacing(fixedImage->GetSpacing());
    gaussImgSource->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
    gaussImgSource->SetMean(point);
    
#if 0
    // If we wanted to find the orientation,  we would use this
    for (int i = 0; i < VDimension; ++i)
      sigma[i] = 3.0 * currScale;
    gaussImgSource->SetSigma(sigma);
#endif
    
    // Simulate the 16x16 Gaussian window descriptor
    // use sigma equal to half the descriptor window width
    for (int i = 0; i < VDimension; ++i)
      sigma[i] = 8.00;
    gaussImgSource->SetSigma(sigma);
    
    gaussImgSource->Update();
    
    IndexType pixelIndex;
    fixedImage->TransformPhysicalPointToIndex(point, pixelIndex);
    
#if 0
    // Only iterate through the region that is within 3 sigma of the mean
    
    IndexType regionStart;
    for (int k=0; k < VDimension; ++k)
      {
	if ((pixelIndex[k] - 3*sigma[k]) > 0)
	  regionStart[k] = (int) floor(pixelIndex[k] - 3*sigma[k]);
	else 
	  regionStart[k] = 0;
      }
      
    typename TFixedImageType::SizeType regionSize = 
      fixedImage->GetLargestPossibleRegion().GetSize();
    
    // Avoid far edge
    for (int k=0; k < VDimension; ++k) {
      if ( ceil(regionStart[k] + 6*sigma[k]) < regionSize[k])
	regionSize[k] =  (int) ceil(6*sigma[k]);
      else
	regionSize[k] -=  regionStart[k];
    }
    
    typename TFixedImageType::RegionType itregion;
    itregion.SetIndex(regionStart);
    itregion.SetSize(regionSize);
#endif
    
    // return the Gaussian weighted Histogram
#ifdef SIFT_FEATURE
    return this->GetSiftKey(hgradImage, gaussImgSource->GetOutput(),
		      pixelIndex);
#else
    return this->GetHistogram(hgradImage, gaussImgSource->GetOutput());
#endif
  }
  
#endif


template <class TFixedImageType, int VDimension> 
  void
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::writeImage(FixedImagePointer fixedImage, const char *filename)
  {
  
  //typedef short      PixelType;
 typedef itk::Image< float, VDimension >    myImageType;
  typedef itk::ImageFileWriter< myImageType >  myWriterType;
  typedef  itk::ResampleImageFilter< TFixedImageType,  myImageType    >    myResampleFilterType;
    
    
  typename myResampleFilterType::Pointer resampler = myResampleFilterType::New();
    
  //resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
  //resampler->SetOutputSpacing( fixedImage->GetSpacing() );
  //resampler->SetOutputOrigin( fixedImage->GetOrigin());
  //resampler->SetTransform(m_IdentityTransform);
  resampler->SetInput(fixedImage);
  resampler->SetReferenceImage(fixedImage);
  resampler->SetUseReferenceImage(true);
  typename myWriterType::Pointer writer = myWriterType::New();
  
 // const char * outputFilename = filename;
  
  writer->SetFileName( filename ); //outputFilename
  //writer->SetInput( fixedImage);
  writer->SetInput( resampler->GetOutput());
  
  std::cout << "[Writing file << " << filename << "]";

   try 
    { 
    writer->Update(); 
    } 
  catch( itk::ExceptionObject & err ) 
    { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    //return EXIT_FAILURE;
    } 
}






  //template <class TFixedImageType, int VDimension> 
  //void
  //ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  //::writeImage(FixedImagePointer fixedImage, const char *filename)
  //{
    
    //if (VDimension != 2) return;
    
    //typedef itk::Image< unsigned char, VDimension >  OutImageType;
    //typedef typename itk::ImageFileWriter< OutImageType  >  FixedWriterType;
    //typedef  itk::ResampleImageFilter< TFixedImageType,  OutImageType    >    
      //OutResampleFilterType;
    
    
    //typename OutResampleFilterType::Pointer resampler = 
      //OutResampleFilterType::New();
    
    //resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
    //resampler->SetOutputSpacing( fixedImage->GetSpacing() );
    //resampler->SetTransform(m_IdentityTransform);
    //resampler->SetInput(fixedImage);
    
  //  typename FixedWriterType::Pointer fixedWriter = FixedWriterType::New();
    
    //fixedWriter->SetFileName(filename);
    //fixedWriter->SetInput( resampler->GetOutput() );
    
    //std::cout << "[Writing file << " << filename << "]";
    
    //try 
   //   {
	//fixedWriter->Update();
      //}
    //catch( itk::ExceptionObject & excep )
      //{
	//std::cerr << "Exception caught !" << std::endl;
	//std::cerr << excep << std::endl;
      //}

  //}
  
  // create a filter that resamples the image (scale up or down) 
  template <class TFixedImageType, int VDimension> 
  typename ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>::ResampleFilterType::Pointer
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::getScaleResampleFilter ( typename TFixedImageType::Pointer fixedImage, float scale )
  {
    typename ResampleFilterType::Pointer scaler = ResampleFilterType::New();
    
    scaler->SetInput( fixedImage );
   	
    // Change the size of the image
    typename TFixedImageType::SizeType size = fixedImage->GetLargestPossibleRegion().GetSize();
    for (int k = 0; k < VDimension; ++k)
      size[k] = (unsigned int) floor(size[k] * scale);
    scaler->SetSize( size );
    //std::cout << " VDIM: " << VDimension << std::endl;
    //std::cout << " NUOVA SIZE: " << size[0] << " " << size[1] << std::endl;

    // Change the spacing
    typename TFixedImageType::SpacingType spacing = fixedImage->GetSpacing();
    for (int k = 0; k < VDimension; ++k)
      spacing[k] = (spacing[k] / scale);
    scaler->SetOutputSpacing( spacing );

    //std::cout << " SPACING AFTER RESAMPLE: " << spacing[0] << " " << spacing[1] << std::endl;
    
    //Origin
    typename TFixedImageType::PointType origin = fixedImage->GetOrigin();
   
   scaler->SetOutputOrigin( fixedImage->GetOrigin() );
   
   //Orientation
   typename TFixedImageType::DirectionType direction;
   direction.SetIdentity();
   direction = fixedImage->GetDirection();
scaler->SetOutputDirection( fixedImage->GetDirection() );

  
        
    //Interpolation
 typedef itk::LinearInterpolateImageFunction< TFixedImageType, double >  InterpolatorType;
 
 typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
 scaler->SetInterpolator( interpolator );
 scaler->SetDefaultPixelValue( 0 );
 scaler->SetTransform( m_IdentityTransform );
 scaler->Update();
    
    return scaler;
  }
  
  template <class TFixedImageType, int VDimension> 
  void
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::SetDoubling (bool tmp) 
  {
    m_DoubleOriginalImage = tmp;
  }

  template <class TFixedImageType, int VDimension>
  void
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::SetNumBins( unsigned int tmp) 
  {
    m_HistogramBinsNumber = tmp;
  }

  template <class TFixedImageType, int VDimension> 
  void
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::SetSigma( double tmp) 
  {
    m_GaussianSigma = tmp;
  }      

  template <class TFixedImageType, int VDimension> 
  void
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::SetNumScales ( unsigned int tmp) 
  {
    m_ImageScalesTestedNumber = tmp;
  }
      
  template <class TFixedImageType, int VDimension> 
  void
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::SetMatchRatio ( float tmp) 
  {
     m_MaxFeatureDistanceRatio = tmp;
  }

// -------------------CURVATURE-----------------

template <class TFixedImageType, int VDimension> 
 unsigned int
 ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
 ::GetHessian(typename myHessianImageType::Pointer ImgInput, IndexType pixelIndex, bool isMax, bool isMin, int i, int j) 
 {
	
	typename HessianGaussianImageFunctionType::TensorType hessian;
	hessian.Fill(0.0);
	hessian=ImgInput->GetPixel(pixelIndex);
	/*std::cout<<"Global Hessian "<<std::endl;
    std::cout<<" ,"<<hessian<<std::endl;*/
	
	FILE* pFileH;
	pFileH=fopen("curve_parameter.txt","a");	
	float Tr=0;
    float Det=1;
    float PrincipalCurve=0;
    float Minors=0;
    float Product=0;
	unsigned int Curvature=0;
    unsigned int P=0;
    unsigned int M=0;
	unsigned int C=0;
	bool maxmin=0;
	if(isMax) maxmin=1;
	if(isMin) maxmin=0;
	
	if(VDimension==2)
	{
	Tr=hessian[0]+hessian[2];
	Det=hessian[0]*hessian[2]-hessian[1]*hessian[1];
	Product=Tr*Det;
    if (Product>0){P=1;}
    PrincipalCurve=(Tr*Tr)/Det; 
    if (PrincipalCurve<m_ThresholdPrincipalCurve){C=1;} 
	Minors=Det;
	if (Minors>0){M=1;}
    }
		
	if(VDimension==3)
	{
	Tr=hessian[0]+hessian[3]+hessian[5];
	Det=hessian[0]*hessian[3]*hessian[5]+2*hessian[1]*hessian[4]*hessian[2]-hessian[0]*hessian[4]*hessian[4]-hessian[3]*(hessian[2])*hessian[2]-hessian[5]*(hessian[1])*hessian[1];
    Product=Tr*Det;

    if (Product>0){P=1;}
    PrincipalCurve=(Tr*Tr*Tr)/Det; 
    if (PrincipalCurve<m_ThresholdPrincipalCurve){C=1;} 
    Minors=hessian[3]*hessian[5]-(hessian[4])*hessian[4]+hessian[5]*hessian[0]-(hessian[2])*hessian[2]+hessian[0]*hessian[3]-(hessian[1])*hessian[1];
    if (Minors>0){M=1;}
    }
    
    if ((Product>0) && (Minors>0)&&(PrincipalCurve<m_ThresholdPrincipalCurve))
	Curvature=1;
	 

	fprintf(pFileH, " %d,%d,%d, %f, %f, %f, %d\n",i,j,maxmin,Product,Minors,PrincipalCurve,Curvature);
	fclose(pFileH);
		/*std::cout << "P:" << Product << " M:" << Minors << " C:" << PrincipalCurve << " cond:" << Curvature << std::endl;
		std::cout << "P:" << P << " M:" << M << " C:" << C << " cond:" << Curvature << std::endl;*/
			
    return Curvature; 
   
 }



//---------- local Hessian ---------------------------

template <class TFixedImageType, int VDimension> 
  unsigned int
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::GetHessianLocal(typename TFixedImageType::Pointer ImgInput, IndexType pixelIndex) 
  {
  
  // Setup operator parameters
  double varianceH =2.0 ;
  varianceH *= varianceH;
  double maxError = 0.001;
  unsigned int maxKernelWidth = 1000;

  // Create function

  typename HessianGaussianImageFunctionType::TensorType hessian;
  hessian.Fill(0.0);
  typename HessianGaussianImageFunctionType::TensorType::EigenValuesArrayType eigenValues;
  eigenValues.Fill(0.0);
  typename HessianGaussianImageFunctionType::Pointer function = HessianGaussianImageFunctionType::New();
  function->SetInputImage( ImgInput );
  function->SetMaximumError( maxError );
  function->SetMaximumKernelWidth( maxKernelWidth );
  function->SetVariance( varianceH );
  function->SetNormalizeAcrossScale( true );
  function->SetUseImageSpacing( true );
  function->SetInterpolationMode( HessianGaussianImageFunctionType::NearestNeighbourInterpolation );
  function->Initialize( );

  hessian = function->EvaluateAtIndex( pixelIndex );
  hessian.ComputeEigenValues( eigenValues );
  
  std::cout<<"Local Hessian "<<std::endl;
  std::cout<<" ,"<<hessian<<std::endl;
      
    float Tr=0;
    float Det=1;
    float PrincipalCurve=0;
    float Minors=0;
    float Product=0;
	unsigned int Curvature=0;
    unsigned int P=0;
    unsigned int M=0;
	unsigned int C=0;
    
    //for( unsigned int i=0; i<VDimension; i++ )
    //{
    //Tr=Tr+eigenValues[i];
    //Det=Det*eigenValues[i];
    //}
    //Product=Tr*Det;
	//	if (Product>0){P=1;}
    
   // for( unsigned int i=0; i<VDimension; i++ )
	//Tr*=Tr;
    
   // PrincipalCurve=Tr/Det;  
	//	if (PrincipalCurve<m_ThresholdPrincipalCurve){C=1;}
    
    //  if (VDimension==2)
	//	{Minors=Det;}
	 // if(VDimension==3)
		//{
		//Minors=eigenValues[0]*eigenValues[1]+eigenValues[1]*eigenValues[2]+eigenValues[0]*eigenValues[2];
		//}
  //  
		//if (Minors>0){M=1;}

	if(VDimension==2)
	{
	Tr=hessian[0]+hessian[2];
	Det=hessian[0]*hessian[2]-hessian[1]*hessian[1];
	Product=Tr*Det;
    if (Product>0){P=1;}
    PrincipalCurve=(Tr*Tr)/Det; 
    if (PrincipalCurve<m_ThresholdPrincipalCurve){C=1;} 
	Minors=Det;
	if (Minors>0){M=1;}
    }
		
	if(VDimension==3)
	{
	Tr=hessian[0]+hessian[3]+hessian[5];
	Det=hessian[0]*hessian[3]*hessian[5]+2*hessian[1]*hessian[4]*hessian[2]-hessian[0]*hessian[4]*hessian[4]-hessian[3]*(hessian[2])*hessian[2]-hessian[5]*(hessian[1])*hessian[1];
    Product=Tr*Det;
    if (Product>0){P=1;}
    PrincipalCurve=(Tr*Tr*Tr)/Det; 
    if (PrincipalCurve<m_ThresholdPrincipalCurve){C=1;} 
    Minors=hessian[3]*hessian[5]-(hessian[4])*hessian[4]+hessian[5]*hessian[0]-(hessian[2])*hessian[2]+hessian[0]*hessian[3]-(hessian[1])*hessian[1];
    if (Minors>0){M=1;}
    }

    
    if ((Product>0) && (Minors>0)){
		Curvature=1;}
	else{
	if(PrincipalCurve<m_ThresholdPrincipalCurve)
	Curvature=1;}
	    
		std::cout << "P:" << Product << " M:" << Minors << " C:" << PrincipalCurve << " cond:" << Curvature << std::endl;
		std::cout << "P:" << P << " M:" << M << " C:" << C << " cond:" << Curvature << std::endl;
		
   //  hessian->DisconnectPipeline();
   //function->ReleaseDataFlagOn(); 
   return Curvature; 
    
  }


//-------------------------------------------------------------


  template <class TFixedImageType, int VDimension> 
  typename ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>::PointSetTypePointer
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::getSiftFeatures(FixedImagePointer fixedImageInput, const char *filename_phy_max, const char *filename_phy_min,const char *filename_im_max, const char *filename_im_min) 
  {
	unsigned int numMin = 0, numMax = 0, numReject = 0;
    m_KeypointSet = PointSetType::New();

    m_PointsCount = 0;  

	// Image normalization [0,1]
	typedef itk::RescaleIntensityImageFilter< TFixedImageType, TFixedImageType > RescaleFilterType;
	typename RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
	rescaleFilter->SetInput(fixedImageInput);
	rescaleFilter->SetOutputMinimum(0);
	rescaleFilter->SetOutputMaximum(1);
 
	typename TFixedImageType::Pointer fixedImage = TFixedImageType::New();
	fixedImage=rescaleFilter->GetOutput();
	//fixedImage->DisconnectPipeline();

	/*char filename[256];
	sprintf(filename, "InputImage.mha");
	this->writeImage(fixedImage, filename);*/


    // Declare Gaussian 
    typedef itk::DiscreteGaussianImageFilter<TFixedImageType, TFixedImageType > GaussianFilterType;
    // Declare DoG 
    typedef itk::SubtractImageFilter<TFixedImageType, TFixedImageType, TFixedImageType> DifferenceFilterType;
    
    //typename GaussianFilterType::ConstPointer gaussianFilter[m_GaussianImagesNumber];
    //std::vector<GaussianFilterType::Pointer> gaussianFilter(m_GaussianImagesNumber);
    //typename DifferenceFilterType::Pointer dogFilter[m_DifferenceOfGaussianImagesNumber];
    //typename TFixedImageType::Pointer dogImage[m_DifferenceOfGaussianImagesNumber];
    //std::vector<DifferenceFilterType::Pointer> dogFilter(m_DifferenceOfGaussianImagesNumber);

#if defined(_MSC_VER)
    //std::vector<typename TFixedImageType::Pointer> gaussianImage_ref;
    std::vector<typename TFixedImageType::Pointer> dogImage(m_DifferenceOfGaussianImagesNumber);
    //std::vector<typename ResampleFilterType::Pointer> scaler(m_ImageScalesTestedNumber);
    //std::vector<typename TFixedImageType::Pointer> scaleImage(m_ImageScalesTestedNumber);
	typename TFixedImageType::Pointer gaussianImage_ref = TFixedImageType::New();

#else
	//typename TFixedImageType::Pointer    gaussianImage(m_GaussianImagesNumber);
    //typename TFixedImageType::Pointer    gaussianImage_ref;
    typename TFixedImageType::Pointer gaussianImage_ref = TFixedImageType::New();

        
    typename TFixedImageType::Pointer	 dogImage[m_DifferenceOfGaussianImagesNumber];
    // Resampled image filters
    //typename ResampleFilterType::Pointer scaler[m_ImageScalesTestedNumber];
    //typename TFixedImageType::Pointer scaleImage[m_ImageScalesTestedNumber];
#endif


#ifdef GENERATE_KEYS

#if defined(_MSC_VER)
    //std::vector<GradientFilterType::Pointer> gradFilter(m_ImageScalesTestedNumber);
    std::vector<typename GradientImageType::Pointer> gradImage(m_ImageScalesTestedNumber);
    std::vector<typename GradientImageType::Pointer> hgradImage(m_ImageScalesTestedNumber);
    //std::vector<typename GradientImageType::Pointer> gradImage;
   

    //std::vector<GradientMagFilterType::Pointer> gradMagFilter(m_ImageScalesTestedNumber);
    std::vector<typename FixedImagePointer> gradMagImage(m_ImageScalesTestedNumber);
#else
    // Declare Gradient
    //typename GradientFilterType::Pointer gradFilter[m_ImageScalesTestedNumber];
    typename GradientImageType::Pointer gradImage[m_ImageScalesTestedNumber];
    typename GradientImageType::Pointer hgradImage[m_ImageScalesTestedNumber];
    //typename GradientImageType::Pointer gradImage;
   
    
    //typename GradientMagFilterType::Pointer gradMagFilter[m_ImageScalesTestedNumber];
    FixedImagePointer gradMagImage[m_ImageScalesTestedNumber];
#endif

#endif


    float currScale = 0.5;

    // For each scale (octave)
    for (unsigned int i = 0; i < m_ImageScalesTestedNumber; ++i) {
      std::cout << "Computing Scale Level (octave) " << i << "... (";

	//typedef itk::Image< PixelType, VDimension > TFixedImageType;
	typename TFixedImageType::Pointer scaleImage = TFixedImageType::New();
	typename ResampleFilterType::Pointer scaler = ResampleFilterType::New();
	typename GaussianFilterType::Pointer tmpGaussianFilter = GaussianFilterType::New();  //this is a new gaussian filter for aliasing correction
    tmpGaussianFilter->ReleaseDataFlagOn();
    
		//if(i>0 )
		//{
		//char filename[256];
	//sprintf(filename, "gauss_ref-%d.mha", i);
	//this->writeImage(gaussianImage_ref, filename);}

      if (i == 0 && !m_DoubleOriginalImage) {

				double variance_anti = 0.5*0.5;
				tmpGaussianFilter->SetVariance(variance_anti);
				tmpGaussianFilter->SetInput( fixedImage );
				//pixel-wise smoothing
				tmpGaussianFilter->SetUseImageSpacing(true); 
				try {
				 tmpGaussianFilter->Update();
					}
				catch( itk::ExceptionObject & excep ) {
				std::cerr << "Exception caught !" << std::endl;
				std::cerr << excep << std::endl;
					}

			scaleImage = tmpGaussianFilter->GetOutput();
			scaleImage->DisconnectPipeline();

		//scaleImage = fixedImage;
		//gaussianImage_ref=fixedImage;
	
      } else {
		 if (i == 0) {
				// Input is the fixed Image.  
	  
				//Antialiasing filter: sigma=0.5;
				//this->writeImage(fixedImage, "pippoIngresso.mha");
			
				//gaussianImage_ref=fixedImage;	 
				

				//double variance = (double)m_SigmaAliasing*m_SigmaAliasing;
				double variance_anti = 0.5*0.5;
				tmpGaussianFilter->SetVariance(variance_anti);
				tmpGaussianFilter->SetInput( fixedImage );
				//pixel-wise smoothing
				tmpGaussianFilter->SetUseImageSpacing(true); 
				try {
				 tmpGaussianFilter->Update();
					}
				catch( itk::ExceptionObject & excep ) {
				std::cerr << "Exception caught !" << std::endl;
				std::cerr << excep << std::endl;
					}

			scaleImage = tmpGaussianFilter->GetOutput();
			scaleImage->DisconnectPipeline();

			scaler = getScaleResampleFilter ( scaleImage, m_ScalingFactor );  //double image
			
			typename GaussianFilterType::Pointer tmpGaussianFilter = GaussianFilterType::New();  //now we need to filter with sigma == 1 because we doubled the size of the image
			
			double variance1 = 1*1;
			tmpGaussianFilter->SetVariance(variance1);
			tmpGaussianFilter->SetInput( scaler->GetOutput() );
			 //pixel-wise smoothing
			tmpGaussianFilter->SetUseImageSpacing(true); 

			try {
				tmpGaussianFilter->Update();
					}
			catch( itk::ExceptionObject & excep ) {
				 std::cerr << "Exception caught !" << std::endl;
				std::cerr << excep << std::endl;
						}

		scaleImage = tmpGaussianFilter->GetOutput();
		scaleImage->DisconnectPipeline();

		char filename[256];
		sprintf(filename, "double_image.mha");
		this->writeImage(scaleImage, filename);

	  
	} else {
	  // Input is the 2*sigma smoothed image from the previous octave
//	  scaler = getScaleResampleFilter ( gaussianImage[m_DifferenceOfGaussianTestsNumber] , 1.0 / m_ScalingFactor );
	  scaler = getScaleResampleFilter ( gaussianImage_ref , 1.0 / m_ScalingFactor );
	  scaleImage = scaler->GetOutput();
	  
	 }
	
	
typename TFixedImageType::SpacingType spacing = scaleImage->GetSpacing();
typename TFixedImageType::PointType origin = scaleImage->GetOrigin();
typename TFixedImageType::DirectionType direction = scaleImage->GetDirection();

//std::cout << " SPACING AFTER ALIAS: " << spacing[0] << " " << spacing[1] <<" "<< spacing[2] <<std::endl;
//std::cout << " ORIGIN AFTER ALIAS: " << origin[0] << " " << origin[1] <<" "<< origin[2] <<std::endl;	
//std::cout << " DIRECTION AFTER ALIAS: " << direction <<std::endl;
      } //else

      {
	typename TFixedImageType::SizeType gsize = 
	  scaleImage->GetLargestPossibleRegion().GetSize();
	for (int j = 0; j < VDimension; ++j)
	  std::cout << gsize[j] << " ";
      }
    
#ifdef DEBUG
      char filename[256];
      sprintf(filename, "gauss-%d-0.png", i);
      this->writeImage(scaleImage, filename);
#endif

#ifdef GENERATE_KEYS
      // ...Compute Gradient
      std::cout << "...Computing Gradient...";
      typename GradientFilterType::Pointer tmpGradFilter = GradientFilterType::New();
      tmpGradFilter->ReleaseDataFlagOn();
      //gradFilter[i] = GradientFilterType::New();
      //gradFilter[i]->SetInput(scaleImage);
      // Do this in pixel space
      //gradFilter[i]->SetUseImageSpacing(true);
      //gradFilter[i]->Update();
      //gradImage[i] = gradFilter[i]->GetOutput();
      //hgradImage[i] = this->GetHypersphericalCoordinates(gradImage[i]);
      tmpGradFilter->SetInput(scaleImage);
      // Do this in pixel space
      tmpGradFilter->SetUseImageSpacing(true);
      tmpGradFilter->Update();
      gradImage[i] = tmpGradFilter->GetOutput();
      hgradImage[i] = this->GetHypersphericalCoordinates(gradImage[i]);
      
      //gradImage = tmpGradFilter->GetOutput();
      
	  typename GradientMagFilterType::Pointer tmpGradMagFilter = GradientMagFilterType::New();
	  tmpGradMagFilter->ReleaseDataFlagOn();
      //gradMagFilter[i] = GradientMagFilterType::New();
      //gradMagFilter[i]->SetInput(scaleImage);
      // Do this in pixel space
      //gradMagFilter[i]->SetUseImageSpacing(true);
      //gradMagFilter[i]->Update();
      //gradMagImage[i] = gradMagFilter[i]->GetOutput();
      tmpGradMagFilter->SetInput(scaleImage);
      // Do this in pixel space
      tmpGradMagFilter->SetUseImageSpacing(true);
      tmpGradMagFilter->Update();
      gradMagImage[i] = tmpGradMagFilter->GetOutput();
      std::cout << "...Done\n";
#endif

typename TFixedImageType::Pointer gaussianImage_old = TFixedImageType::New();
	typename TFixedImageType::Pointer gaussianImage_new = TFixedImageType::New();
	typename TFixedImageType::Pointer gaussianImage_start = TFixedImageType::New();

      // ...Compute Gaussians
      for (unsigned int j = 0; j < m_GaussianImagesNumber; ++j) {
#ifdef VERBOSE
	std::cout << "Setting Up Gaussian Filter " << i << "-" << j << "...";
	std::cout.flush();
#endif
	
	if (j==0)
	{
		typename GaussianFilterType::Pointer tmpGaussianFilter = GaussianFilterType::New();

		/* Variance is square of the sigma
		* sigma = (2^(j/s)*sigma)
		*/

		double variance = this->GetGaussianScale(j);
		std::cout<<"scala: "<<j<<" sigma: "<<variance<<std::endl;
		variance *= variance;
		//gaussianFilter[j]->SetVariance(variance);
		//gaussianFilter[j]->SetInput( scaleImage );
		// pixel-wise smoothing
		//gaussianFilter[j]->SetUseImageSpacing(true); 
		//try {
		//  gaussianFilter[j]->Update();
		//}
		tmpGaussianFilter->ReleaseDataFlagOn();
		tmpGaussianFilter->SetVariance(variance);
		tmpGaussianFilter->SetInput( scaleImage );
		// pixel-wise smoothing
		tmpGaussianFilter->SetUseImageSpacing(true); 
		try {
			tmpGaussianFilter->Update();
			}
		catch( itk::ExceptionObject & excep ) {
			 std::cerr << "Exception caught !" << std::endl;
			 std::cerr << excep << std::endl;
				}

		//gaussianImage[j] = tmpGaussianFilter->GetOutput();
		gaussianImage_start= tmpGaussianFilter->GetOutput();
		gaussianImage_start->DisconnectPipeline();
		//tmpGaussianFilter->ReleaseDataFlagOff();
	}
	
	
	if(j>0)
	{	
		if(j==1)
		gaussianImage_old=gaussianImage_start;
		
		typename GaussianFilterType::Pointer tmpGaussianFilter = GaussianFilterType::New();

		/* Variance is square of the sigma
		 * sigma = (2^(j/s)*sigma)
		*/

		double variance = this->GetGaussianScale(j);
		std::cout<<"scala: "<<j<<" sigma: "<<variance<<std::endl;
		
		variance *= variance;
		//gaussianFilter[j]->SetVariance(variance);
		//gaussianFilter[j]->SetInput( scaleImage );
		// pixel-wise smoothing
		//gaussianFilter[j]->SetUseImageSpacing(true); 
		//try {
		//  gaussianFilter[j]->Update();
		//}
		tmpGaussianFilter->ReleaseDataFlagOn();
		tmpGaussianFilter->SetVariance(variance);
		tmpGaussianFilter->SetInput( scaleImage );
		// pixel-wise smoothing
		tmpGaussianFilter->SetUseImageSpacing(true); 
		try {
			tmpGaussianFilter->Update();
			}
		catch( itk::ExceptionObject & excep ) {
			std::cerr << "Exception caught !" << std::endl;
			std::cerr << excep << std::endl;
			}

		//gaussianImage[j] = tmpGaussianFilter->GetOutput();
		gaussianImage_new = tmpGaussianFilter->GetOutput();
		gaussianImage_new->DisconnectPipeline();
		//tmpGaussianFilter->ReleaseDataFlagOff();
	
		if(j==m_DifferenceOfGaussianTestsNumber)
		{
		gaussianImage_ref=gaussianImage_new;
		}

		//typename TFixedImageType::SpacingType spacing = gaussianImage[j]->GetSpacing();
		typename TFixedImageType::SpacingType spacing = gaussianImage_new->GetSpacing();
	
#ifdef DEBUG
	char filename[256];
	sprintf(filename, "gauss-%d-%d.png", i, j);
	//this->writeImage(gaussianImage[j], filename);
	this->writeImage(gaussianImage_new, filename);
#endif

#ifdef VERBOSE
	std::cout << "Done\n";
	std::cout.flush();
#endif
   //   }
    
      // ...Compute Difference of Gaussians
     // for (unsigned int j = 0; j < (m_DifferenceOfGaussianImagesNumber); ++j) 
     //{
#ifdef VERBOSE
	std::cout << "Setting Up DoG Filter " << i << "-" << j << "...";
	std::cout.flush();
#endif

		if (j<m_DifferenceOfGaussianImagesNumber+1)
		{
			typename DifferenceFilterType::Pointer tmpDogFilter = DifferenceFilterType::New();
			//tmpDogFilter->SetInput1( gaussianImage[j] );
			//tmpDogFilter->SetInput2( gaussianImage[j+1] );
			tmpDogFilter->ReleaseDataFlagOn();
			tmpDogFilter->SetInput1( gaussianImage_old );
			tmpDogFilter->SetInput2( gaussianImage_new );
			tmpDogFilter->Update();	
			
			dogImage[j-1] = tmpDogFilter->GetOutput();
			//tmpDogFilter->ReleaseDataFlagOff();
		
			typename TFixedImageType::SpacingType spacing = dogImage[j-1]->GetSpacing();
			//std::cout << " SPACING DOG IMAGE: " << spacing[0] << " " << spacing[1] << std::endl;
	
			//dogFilter[j] = DifferenceFilterType::New();
			//dogFilter[j]->SetInput1( gaussianImage_old );
			//dogFilter[j]->SetInput2( gaussianImage_new );
			//dogFilter[j]->Update();
			//dogImage[j] = dogFilter[j]->GetOutput();
	
			char filename[256];
			sprintf(filename, "dog-%d-%d.mha", i, j-1);
			this->writeImage(dogImage[j-1], filename);
	
			}
	
#ifdef DEBUG
	char filename[256];
	sprintf(filename, "dog-%d-%d.png", i, j-1);
	this->writeImage(dogImage[j], filename);
#endif

#ifdef VERBOSE
	std::cout << "Done\n";
	std::cout.flush();
#endif 
    //  }
    
    //char filename[256];
	//sprintf(filename, "gauss-%d-%d.mha", i, j);
	//this->writeImage(gaussianImage_new, filename);
	
    gaussianImage_old=gaussianImage_new; //update
    
    } //if (j>=0)
    
   } //for gaussian


      for (unsigned int j=1; j < (m_DifferenceOfGaussianImagesNumber - 1); ++j) {
	// Search the dogImages for local maxima,  w.r.t. corresponding
	// point in the scale above and below
	// level 0 is the "doubled" image
	// Iterate over the various doG filters
	// Only use the middle dogs (ones with both neighbours above and below)
	// Iterate over each position in the dog filter
	typedef itk::ImageRegionIteratorWithIndex< TFixedImageType > 
	  ImageIteratorType;

	IndexType regionStart;
	// Avoid the edges
	for (int k=0; k < VDimension; ++k)
	  regionStart[k] = 1;


	typename TFixedImageType::SizeType regionSize = 
	  dogImage[j]->GetLargestPossibleRegion().GetSize();
	  
	//----------Hessian----------
    typedef itk::HessianRecursiveGaussianImageFilter< TFixedImageType >  myFilterType;
    typedef typename myFilterType::OutputImageType myHessianImageType;
    typename myHessianImageType::Pointer HessianImage = myHessianImageType::New();
    typename myFilterType::Pointer filter_hess = myFilterType::New();
    filter_hess->SetInput( dogImage[j] ); 
    filter_hess->SetSigma( 2.0 ); 
    filter_hess->Update();
    HessianImage = filter_hess->GetOutput();
      
    HessianImage->DisconnectPipeline();
    filter_hess->ReleaseDataFlagOn();  
    
    
    

#ifdef VERBOSE
	std::cout << "Searching for Extrema in DoG Image " << i << "-" << j;
	std::cout << " ( ";
	for (int k=0; k < VDimension; ++k)
	  std::cout << regionSize[k] << " ";
	std::cout << ") Scale " << currScale << "\n";
	std::cout.flush();
#endif

	// Avoid far edge
	for (int k=0; k < VDimension; ++k)
	  regionSize[k] -=  2;
	  
	typename TFixedImageType::RegionType itregion;
	itregion.SetIndex(regionStart);
	itregion.SetSize(regionSize);
	      
	ImageIteratorType pixelIt(dogImage[j],
				  itregion);
      
      // this iterates on the pixels
	for ( pixelIt.GoToBegin(); !pixelIt.IsAtEnd(); ++pixelIt) {
	  // Make sure to start sufficiently into the image so that all
	  // neighbours are present
	  IndexType pixelIndex = pixelIt.GetIndex();
	  typename TFixedImageType::PixelType pixelValue = pixelIt.Get();
	PointType point;
	//typename TFixedImageType::SpacingType spacing;
	//typename TFixedImageType::OriginType origin;
	PointType vertex;
	FILE* pFile;
	FILE* pFile1;
	FILE* pFileRejectContrast;
	FILE* pFileRejectCurvature;
	dogImage[j]->TransformIndexToPhysicalPoint (pixelIndex, point); 

	  
#ifdef ERROR_CHECK
	  std::cerr << "Checking ( ";
	  for (int k = 0; k < VDimension; ++k)
	    std::cerr << pixelIndex[k] << " ";
	  std::cerr << ") = " << pixelValue <<"\n";
#endif

	  // Compare to the 8 immediate neighbours
	  bool isMax=true;
	  bool isMin=true;

	  this->CheckLocalExtrema(dogImage[j], 
			    pixelIndex, pixelValue, isMax, isMin, false);

	  if (!isMax && !isMin) continue;
	  
	  // Compare to scale above
	  if (j < (m_GaussianImagesNumber-1)) {
#ifdef DEBUG_VERBOSE
	    std::cout << "...Checking Scale Above\n";
#endif
	    //dogImage[i+1][j]->TransformPhysicalPointToIndex (point, tmpIndex);

	    this->CheckLocalExtrema(dogImage[j+1], 
			      pixelIndex, pixelValue, isMax, isMin, true);
	  }
	  if (!isMax && !isMin) continue;

	  // Compare to scale below
	  if (j > 0) {
#ifdef DEBUG_VERBOSE
	    std::cout << "...Checking Scale Below\n";
#endif
	    //dogImage[i-1][j]->TransformPhysicalPointToIndex (point, tmpIndex);

	    this->CheckLocalExtrema(dogImage[j-1], 
			      pixelIndex, pixelValue, isMax, isMin, true);
	  }
	
	  /*if(isMax) 
	  std::cout<< "MAX:" <<std::endl;
	  if(isMin)
	  std:: cout<< "min:" <<std::endl;*/
	  
	  if (!isMax && !isMin) continue;
	  
	  // Check if it is sufficiently large (absolute value) - thresholding on image contrast
	  
	  if (fabs(pixelValue) < m_MinKeypointValue) {
		  //std::cout<< "modulo: "<<fabs(pixelValue)<<std::endl;
	    ++numReject;
		pFileRejectContrast=fopen("point_rejected_contrast.fcsv","a");
		
		point[0]=-1.0*point[0];
		point[1]=-1.0*point[1];

		if(isMax){
			fprintf(pFileRejectContrast,"M");
			}
		if(isMin){
			fprintf(pFileRejectContrast,"m");
			}
		fprintf(pFileRejectContrast, "-%d-%d-%d,",numReject,i,j);
		for(int k=0; k<VDimension; k++)
	  	  {
	  		fprintf(pFileRejectContrast,"%.3f, ",point[k]);
	  		} 	  
	  		fprintf(pFileRejectContrast,"\n");
	  		fclose(pFileRejectContrast);

	    continue;
	  }
	  
	 // Thresholding on image curvature
	unsigned int Curvature=0;

	Curvature=this->GetHessian(HessianImage, pixelIndex, isMax, isMin, i, j);
	//Curvature=this->GetHessianLocal(dogImage[j],pixelIndex);
	 
	if ( Curvature==0 ){
		++numReject;

		pFileRejectCurvature=fopen("point_rejected_curvature.fcsv","a");
		
		point[0]=-1.0*point[0];
		point[1]=-1.0*point[1];

		if(isMax){
			fprintf(pFileRejectCurvature,"M");
			}
		if(isMin){
			fprintf(pFileRejectCurvature,"m");
			}	
		fprintf(pFileRejectCurvature, "-%d-%d-%d,",numReject,i,j);
		for(int k=0; k<VDimension; k++)
	  	  {
	  		fprintf(pFileRejectCurvature,"%.3f, ",point[k]);
	  		} 	  
	  		fprintf(pFileRejectCurvature,"\n");
	  		fclose(pFileRejectCurvature);

	   continue;
	  }
	 
	  

	  // Passed all checks:

#ifdef DEBUG
	  std::cout << point << std::endl;
#endif
	  m_KeypointSet->SetPoint( m_PointsCount, point);
#ifdef GENERATE_KEYS
	  // Generate features
	  // Space used is the (smoothed) original image)
	 //m_KeypointSet->SetPointData( m_PointsCount, 
	//			   this->GetFeatures( gaussianImage[0], 
						//      hgradImage[i], point,
						//      this->GetGaussianScale(j)));
	 m_KeypointSet->SetPointData( m_PointsCount, 
				   this->GetFeatures( gaussianImage_start, 
						      hgradImage[i], point,
						      this->GetGaussianScale(j)));
#else
	  m_KeypointSet->SetPointData( m_PointsCount, currScale);
#endif
	  ++m_PointsCount;

	  if (isMax) {
	    // Maxima detected.  
	    ++numMax;
	    //std::cout << "max phys coord: "<< point << std::endl;
	   //std::cout << "max image coord: "<< pixelIndex << std::endl;
	      
	    pFile=fopen(filename_phy_max,"a");
		pFile1=fopen(filename_im_max,"a");
		
		point[0]=-1.0*point[0];
		point[1]=-1.0*point[1];
		
		fprintf(pFile, "M-%d-%d-%d,", numMax,i,j);
		for(int k=0; k<VDimension; k++)
	  	  {
	  		
	  		fprintf(pFile,"%.3f, ",point[k]);
	  		} 	  
	  		fprintf(pFile,"\n");
	  		fclose(pFile);
	  		
	  	point[0]=-1.0*point[0];
		point[1]=-1.0*point[1];
	  		 	  
	  	  fprintf(pFile1, "%d,", numMax);
	  	  for(int k=0; k<VDimension; k++)
	  	  {
	  		 vertex[k]= (point[k]-fixedImage->GetOrigin()[k])/fixedImage->GetSpacing()[k];
	  		 fprintf(pFile1,"%.3f, ",vertex[k]);
		  }
		fprintf(pFile1,"\n");
		fclose(pFile1);
	   
#ifdef DEBUG
	    std::cout << "Found Maxima! ";
#endif
	  }
	  if (isMin) {
	    // Minima detected.  
	    ++numMin;
	    //std::cout << "min phys coord: "<< point << std::endl;
	    //std::cout << "min image coord: "<< pixelIndex << std::endl;
	   	    
	    pFile=fopen(filename_phy_min,"a");
		pFile1=fopen(filename_im_min,"a");
		
		point[0]=-1.0*point[0];
		point[1]=-1.0*point[1];
		
		fprintf(pFile, "m-%d-%d-%d,", numMin,i,j);
		for(int k=0; k<VDimension; k++)
	  	  {
	  		fprintf(pFile,"%.3f, ",point[k]);
	  		} 	  
	  		fprintf(pFile,"\n");
	  		fclose(pFile);
	  	
	  	point[0]=-1.0*point[0];
		point[1]=-1.0*point[1];
	  		  		
	  	  fprintf(pFile1, "%d,", numMin);
	  	  for(int k=0; k<VDimension; k++)
	  	  {
	  		 
	  		 vertex[k]= (point[k]-fixedImage->GetOrigin()[k])/fixedImage->GetSpacing()[k];
	  		 fprintf(pFile1,"%.3f, ",vertex[k]);
		  }
		fprintf(pFile1,"\n");
		fclose(pFile1);

#ifdef DEBUG
	    std::cout << "Found Minima! ";
#endif
	  }
	  //std::cout << "current scale: "<< currScale << std::endl;
	}
#ifdef VERBOSE
	std::cout << "Acc. Num Max: " << numMax 
		  << "\nAcc. Num Min: " << numMin 
		  << "\nAcc. Num Reject: " << numReject 
		  << std::endl;
	std::cout.flush();
#endif
      }
      currScale *= m_ScalingFactor;
    }
	
#ifdef VERBOSE
    std::cout << "Total Num Max: " << numMax 
	      << "\nTotal Num Min: " << numMin 
	      << "\nTotal Num Reject: " << numReject
	      << std::endl;	
    std::cout.flush();
#endif
    return m_KeypointSet;


  }


	  	  
	




  template <class TFixedImageType, int VDimension> 
  void
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::MatchKeypointsPos(PointSetTypePointer keypoints1, PointSetTypePointer keypoints2,
		      typename TransformType::Pointer inverse_transform)
  {
    // Compare Keypoints.  Check Coverage and Accuracy
    // This does the comparison based on position of the keypoints
    // Find:  
    // # of points that match which will tell us
    // # of points that did not scale
    // # of points created by the scale
    unsigned int numMatches;
    unsigned int numMatches2;
    unsigned int numMatches5;
    const double MATCH_THRESHOLD = 1.5;
    typename PointSetType::PointsContainer::Pointer keyps1, keyps2;

    unsigned long numpoints1, numpoints2;
    numpoints1 = keypoints1->GetNumberOfPoints();
    std::cout << "Keypoints1 Found: " << numpoints1 << std::endl;
    numpoints2 = keypoints2->GetNumberOfPoints();
    std::cout << "Keypoints2 Found: " << numpoints2 << std::endl;

    if (!inverse_transform)
      return;

    numMatches = 0;
    numMatches2 = 0;
    numMatches5 = 0;
    for (unsigned int i = 0; i < numpoints2; ++i) {
      PointType pp2;
      pp2.Fill(0.0);
      keypoints2->GetPoint(i, &pp2);	
	
      bool match = false;
      bool match2 = false;
      bool match5 = false;
      for (unsigned int j = 0; j < numpoints1; ++j) {
	PointType tmpp, pp;
	keypoints1->GetPoint(j, &tmpp);
	pp = inverse_transform->TransformPoint(tmpp);
	//	    for (int k = 0; k < VDimension; ++k)
	//	      pp[k] *= scale_test;	      
	if(!match  && pp.EuclideanDistanceTo(pp2) <= MATCH_THRESHOLD)
	  {
	    ++numMatches;
	    match = true;
	  }
	if(!match2 && pp.EuclideanDistanceTo(pp2) <= 2*MATCH_THRESHOLD)
	  {
	    ++numMatches2;
	    match2 = true;
	  }
	if(!match5 && pp.EuclideanDistanceTo(pp2) <= 5*MATCH_THRESHOLD)
	  {
	    ++numMatches5;
	    match5 = true;
	  }
	if (match && match2 && match5)
	  break;
      }      
    }

    std::cout << "Keypoints 2 Matching to Keypoints 1 (<" << MATCH_THRESHOLD << "): " << numMatches << std::endl;
    std::cout << "% of Keypoints 2 Matching (<" << MATCH_THRESHOLD << "):  " << (float) numMatches / numpoints2 << std::endl;
    std::cout << "Keypoints 2 Matching to Keypoints 1 (<" << 2*MATCH_THRESHOLD << "): " << numMatches2 << std::endl;
    std::cout << "% of Keypoints 2 Matching (<" << 2*MATCH_THRESHOLD << "):  " << (float) numMatches2 / numpoints2 << std::endl;
    std::cout << "Keypoints 2 Matching to Keypoints 1 (<" << 5*MATCH_THRESHOLD << "): " << numMatches5 << std::endl;
    std::cout << "% of Keypoints 2 Matching (<" << 5*MATCH_THRESHOLD << "):  " << (float) numMatches5 / numpoints2 << std::endl;


    numMatches = 0;
    numMatches2 = 0;
    numMatches5 = 0;
    for (unsigned int j = 0; j < numpoints1; ++j) {
      PointType tmpp, pp;
      keypoints1->GetPoint(j, &tmpp);
      pp = inverse_transform->TransformPoint(tmpp);
      //	  for (int k = 0; k < VDimension; ++k)
      //	    pp[k] *= scale_test;

      bool match = false;
      bool match2 = false;
      bool match5 = false;	
      for (unsigned int i = 0; i < numpoints2; ++i) {
	PointType pp2;
	pp2.Fill(0.0);
	keypoints2->GetPoint(i, &pp2);	

	if(!match  && pp.EuclideanDistanceTo(pp2) <= MATCH_THRESHOLD)
	  {
	    ++numMatches;
	    match = true;
	  }
	if(!match2 && pp.EuclideanDistanceTo(pp2) <= 2*MATCH_THRESHOLD)
	  {
	    ++numMatches2;
	    match2 = true;
	  }
	if(!match5 && pp.EuclideanDistanceTo(pp2) <= 5*MATCH_THRESHOLD)
	  {
	    ++numMatches5;
	    match5 = true;
	  }
	if (match && match2 && match5)
	  break;
      }      
    }

    std::cout << "Keypoints 1 Matching to Keypoints 2 (<" << MATCH_THRESHOLD << "): " << numMatches << std::endl;
    std::cout << "% of Keypoints 1 Matching (<" << MATCH_THRESHOLD << "):  " << (float) numMatches / numpoints1 << std::endl;
    std::cout << "Keypoints 1 Matching to Keypoints 2 (<" << 2*MATCH_THRESHOLD << "): " << numMatches2 << std::endl;
    std::cout << "% of Keypoints 1 Matching (<" << 2*MATCH_THRESHOLD << "):  " << (float) numMatches2 / numpoints1 << std::endl;
    std::cout << "Keypoints 1 Matching to Keypoints 2 (<" << 5*MATCH_THRESHOLD << "): " << numMatches5 << std::endl;
    std::cout << "% of Keypoints 1 Matching (<" << 5*MATCH_THRESHOLD << "):  " << (float) numMatches5 / numpoints1 << std::endl;
  }

#ifdef GENERATE_KEYS
  template <class TFixedImageType, int VDimension> 
  void
  ScaleInvariantFeatureImageFilter<TFixedImageType,VDimension>
  ::MatchKeypointsFeatures(PointSetTypePointer keypoints1, PointSetTypePointer keypoints2,
			   typename TransformType::Pointer inverse_transform)
  {
    // Compare Keypoints.  Check Coverage and Accuracy
    // This does the comparison based on position of the keypoints
    // Find:  
    // # of points that match which will tell us
    // # of points that did not scale
    // # of points created by the scale
    unsigned int numMatches;
    unsigned int numMatchesTried;
    unsigned int numMatches2;
    unsigned int numMatches5;
    const double MATCH_THRESHOLD = 1.5;
    typename PointSetType::PointsContainer::Pointer keyps1, keyps2;

    unsigned long numpoints1, numpoints2;
    numpoints1 = keypoints1->GetNumberOfPoints();
    std::cout << "Keypoints1 Found: " << numpoints1 << std::endl;
    numpoints2 = keypoints2->GetNumberOfPoints();
    std::cout << "Keypoints2 Found: " << numpoints2 << std::endl;

    std::cout << "***Keypoint Matches***\n";

    numMatches = 0;
    numMatches2 = 0;
    numMatches5 = 0;
    numMatchesTried = 0;
    for (unsigned int i = 0; i < numpoints2; ++i) {
      PointType pp2;
      pp2.Fill(0.0);
      keypoints2->GetPoint(i, &pp2);	
      FeatureType ft2;
      keypoints2->GetPointData(i, &ft2);	
	
      FeatureType bestft;
      float bestdist = -1.0;
      float nextbestdist = -1.0;
      unsigned int bestj=0;
      for (unsigned int j = 0; j < numpoints1; ++j) {
	PointType pp;
	keypoints1->GetPoint(j, &pp);
	FeatureType ft;
	keypoints1->GetPointData(j, &ft);	

	float dist = 0.0;
	for (unsigned int k = 0; k < ft.Size(); ++k)
	  dist += (ft[k] - ft2[k])*(ft[k] - ft2[k]);
	  
	if (nextbestdist < 0.0 || dist < bestdist)
	  {
	    nextbestdist = bestdist;
	    bestdist=dist;
	    bestft = ft;
	    bestj = j;
	  }	  
      }

      /* Reject "too close" matches */
      if ((bestdist / nextbestdist) >  m_MaxFeatureDistanceRatio)
	continue;	

      /* NEW IDEA -- look to make sure it is a reciprocal best match */
      /* Take the best feature found,  see if pp2 makes the cut */
      bestdist = -1.0;
      nextbestdist = -1.0;
      FeatureType bestft2;
      unsigned int bestj2;

      for (unsigned int j = 0; j < numpoints2; ++j) {
	PointType pp;
	keypoints2->GetPoint(j, &pp);
	FeatureType ft;
	keypoints2->GetPointData(j, &ft);	

	float dist = 0.0;
	for (unsigned int k = 0; k < ft.Size(); ++k)
	  dist += (ft[k] - bestft[k])*(ft[k] - bestft[k]);
	  
	if (nextbestdist < 0.0 || dist < bestdist)
	  {
	    nextbestdist = bestdist;
	    bestdist=dist;
	    bestft2 = ft;
	    bestj2 = j;
	  }	  
      }

      /* Reject if not reciprocal best hit or "too close" matches */
      if ( bestft2 != ft2 || ((bestdist / nextbestdist) >  m_MaxFeatureDistanceRatio))
	continue;	
      /* END NEW IDEA */

      ++numMatchesTried;

      // Check goodness of best feature
      PointType tmpp, pp;
      keypoints1->GetPoint(bestj, &tmpp);

      // Print the match
      std::cout << tmpp << " => " << pp2 << std::endl;

      if (!inverse_transform)
	continue;
      pp = inverse_transform->TransformPoint(tmpp);

      if(pp.EuclideanDistanceTo(pp2) <= MATCH_THRESHOLD)
	++numMatches;
      if(pp.EuclideanDistanceTo(pp2) <= (2*MATCH_THRESHOLD))
	++numMatches2;
      if(pp.EuclideanDistanceTo(pp2) <= (5*MATCH_THRESHOLD))
	++numMatches5;
    }

    std::cout << "\n***Features 2 Matches Attempted: " << numMatchesTried << std::endl;
    std::cout << "Features 2 Matching to Features 1 (<" << MATCH_THRESHOLD << "): " << numMatches << std::endl;
    std::cout << "% of Features 2 Matching (<" << MATCH_THRESHOLD << "):  " << (float) numMatches / numMatchesTried << std::endl;
    std::cout << "Features 2 Matching to Features 1 (<" << 2*MATCH_THRESHOLD << "): " << numMatches2 << std::endl;
    std::cout << "% of Features 2 Matching (<" << 2*MATCH_THRESHOLD << "):  " << (float) numMatches2 / numMatchesTried << std::endl;
    std::cout << "Features 2 Matching to Features 1 (<" << 5*MATCH_THRESHOLD << "): " << numMatches5 << std::endl;
    std::cout << "% of Features 2 Matching (<" << 5*MATCH_THRESHOLD << "):  " << (float) numMatches5 / numMatchesTried << std::endl;

  }
#endif

} // end namespace itk

#endif
