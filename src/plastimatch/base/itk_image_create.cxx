/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include "itkImage.h"

#include "itk_image_type.h"
#include "itk_image_create.h"
#include "plm_image_header.h"

/* -----------------------------------------------------------------------
   Create a new image
   ----------------------------------------------------------------------- */
template<class T> 
typename itk::Image<T,3>::Pointer
itk_image_create (const Plm_image_header& pih)
{
#if defined (commentout)
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


#endif

    typename itk::Image<T,3>::Pointer img = itk::Image<T,3>::New ();
    img->SetOrigin (pih.GetOrigin());
    img->SetSpacing (pih.GetSpacing());
    img->SetDirection (pih.GetDirection());
    img->SetRegions (pih.GetLargestPossibleRegion());
    img->Allocate ();
    img->FillBuffer (static_cast<T>(0));

    return img;

#if defined (commentout)
    d_ptr->weights = FloatImageType::New();
    d_ptr->weights->SetOrigin (d_ptr->target->GetOrigin());
    d_ptr->weights->SetSpacing (d_ptr->target->GetSpacing());
    d_ptr->weights->SetDirection (d_ptr->target->GetDirection());
    d_ptr->weights->SetRegions (d_ptr->target->GetLargestPossibleRegion());
    d_ptr->weights->Allocate ();
    d_ptr->weights->FillBuffer (0.0);
#endif
}

/* Explicit instantiations */
template PLMBASE_API itk::Image<float,3>::Pointer itk_image_create<float> (const Plm_image_header& pih);
//template PLMBASE_API void itk_image_stats (ShortImageType::Pointer, double*, double*, double*, int*, int*);
//template PLMBASE_API void itk_image_stats (FloatImageType::Pointer, double*, double*, double*, int*, int*);
