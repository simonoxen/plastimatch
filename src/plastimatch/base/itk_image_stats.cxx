/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include "itkImage.h"
#include "itkOrientImageFilter.h"

#include "image_stats.h"
#include "itk_image_type.h"
#include "itk_image_stats.h"

/* -----------------------------------------------------------------------
   Statistics like min, max, etc.
   ----------------------------------------------------------------------- */
template<class T> 
void
itk_image_stats (const T& img, Image_stats *image_stats)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ImageRegionIterator< ImageType > IteratorType;
    typename ImageType::RegionType rg = img->GetLargestPossibleRegion ();
    IteratorType it (img, rg);

    int first = 1;
    double sum = 0.0;

    image_stats->num_non_zero = 0;
    image_stats->num_vox = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	double v = (double) it.Get();
	if (first) {
	    image_stats->min_val = image_stats->max_val = v;
	    first = 0;
	}
	if (image_stats->min_val > v) image_stats->min_val = v;
	if (image_stats->max_val < v) image_stats->max_val = v;
	sum += v;
	image_stats->num_vox ++;
	if (v != 0.0) {
	    image_stats->num_non_zero ++;
	}
    }
    image_stats->avg_val = sum / image_stats->num_vox;
}

template<class T> 
void
itk_image_stats (T img, double *min_val, double *max_val, 
    double *avg, int *non_zero, int *num_vox)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ImageRegionIterator< ImageType > IteratorType;
    typename ImageType::RegionType rg = img->GetLargestPossibleRegion ();
    IteratorType it (img, rg);

    int first = 1;
    double sum = 0.0;

    *non_zero = 0;
    *num_vox = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	double v = (double) it.Get();
	if (first) {
	    *min_val = *max_val = v;
	    first = 0;
	}
	if (*min_val > v) *min_val = v;
	if (*max_val < v) *max_val = v;
	sum += v;
	(*num_vox) ++;
	if (v != 0.0) {
	    (*non_zero) ++;
	}
    }
    *avg = sum / (*num_vox);
}

/* Explicit instantiations */
template PLMBASE_API void itk_image_stats (UCharImageType::Pointer, double*, double*, double*, int*, int*);
template PLMBASE_API void itk_image_stats (ShortImageType::Pointer, double*, double*, double*, int*, int*);
template PLMBASE_API void itk_image_stats (Int32ImageType::Pointer, double*, double*, double*, int*, int*);
template PLMBASE_API void itk_image_stats (FloatImageType::Pointer, double*, double*, double*, int*, int*);

template PLMBASE_API void itk_image_stats (const UCharImageType::Pointer&, Image_stats*);
template PLMBASE_API void itk_image_stats (const ShortImageType::Pointer&, Image_stats*);
template PLMBASE_API void itk_image_stats (const UShortImageType::Pointer&, Image_stats*);
template PLMBASE_API void itk_image_stats (const Int32ImageType::Pointer&, Image_stats*);
template PLMBASE_API void itk_image_stats (const FloatImageType::Pointer&, Image_stats*);

