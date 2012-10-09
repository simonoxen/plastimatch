/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <time.h>
#include "itkSubtractImageFilter.h"
#include "itkImageRegionIterator.h"

#include "diff.h"
#include "itk_image_save.h"
#include "plm_image.h"
#include "print_and_exit.h"

Diff_parms::Diff_parms ()
{
}

void
diff_main (Diff_parms* parms)
{
    Plm_image *img1, *img2;

    img1 = plm_image_load_native ((const char*) parms->img_in_1_fn);
    if (!img1) {
	print_and_exit ("Error: could not open '%s' for read\n",
	    (const char*) parms->img_in_1_fn);
    }
    img2 = plm_image_load_native ((const char*) parms->img_in_2_fn);
    if (!img2) {
	print_and_exit ("Error: could not open '%s' for read\n",
	    (const char*) parms->img_in_2_fn);
    }

#if defined (commentout)
    /* Debugging Nami's problem */
    printf ("Img type: %s, %s\n",
	plm_image_type_string (img1->m_type),
	plm_image_type_string (img2->m_type));
#endif

#if defined (commentout)
    /* Debugging Nami's problem */
    if (img1->m_type == PLM_IMG_TYPE_ITK_SHORT) {
	double min_val, max_val, avg;
	int non_zero, num_vox;
	itk_image_stats (img1->m_itk_short, 
	    &min_val, &max_val, &avg, &non_zero, &num_vox);
	printf ("MIN %f AVE %f MAX %f NONZERO %d NUMVOX %d\n", 
	    (float) min_val, (float) avg, (float) max_val, non_zero, num_vox);

    }
#endif

    if (!Plm_image::compare_headers (img1, img2)) {
	print_and_exit ("Error: image sizes do not match\n");
    }

    FloatImageType::Pointer fi1 = img1->itk_float ();
    FloatImageType::Pointer fi2 = img2->itk_float ();

#if defined (commentout)
    {
	/* Debugging Nami's problem */
	double min_val, max_val, avg;
	int non_zero, num_vox;
	itk_image_stats (fi1,
	    &min_val, &max_val, &avg, &non_zero, &num_vox);
	printf ("MIN %f AVE %f MAX %f NONZERO %d NUMVOX %d\n", 
	    (float) min_val, (float) avg, (float) max_val, non_zero, num_vox);
	itk_image_stats (fi2,
	    &min_val, &max_val, &avg, &non_zero, &num_vox);
	printf ("MIN %f AVE %f MAX %f NONZERO %d NUMVOX %d\n", 
	    (float) min_val, (float) avg, (float) max_val, non_zero, num_vox);
    }
#endif

    typedef itk::SubtractImageFilter< FloatImageType, FloatImageType, 
				      FloatImageType > SubtractFilterType;
    SubtractFilterType::Pointer sub_filter = SubtractFilterType::New();

    sub_filter->SetInput1 (fi1);
    sub_filter->SetInput2 (fi2);

    try {
	sub_filter->Update();
    } catch (itk::ExceptionObject & excep) {
	std::cerr << "ITK exception caught: " << excep << std::endl;
	exit (-1);
    }
    FloatImageType::Pointer diff = sub_filter->GetOutput ();

    itk_image_save_float (diff, (const char*) parms->img_out_fn);
}

