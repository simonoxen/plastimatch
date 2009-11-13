/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkSubtractImageFilter.h"
#include "itkImageRegionIterator.h"
#include "getopt.h"
#include "compare_main.h"
#include "plm_image.h"
#include "itk_image.h"

static void
compare_main (Compare_parms* parms)
{
    PlmImage *img1, *img2;

    img1 = plm_image_load_native (parms->img_in_1_fn);
    if (!img1) {
	print_and_exit ("Error: could not open '%s' for read\n",
		       parms->img_in_1_fn);
    }
    img2 = plm_image_load_native (parms->img_in_2_fn);
    if (!img2) {
	print_and_exit ("Error: could not open '%s' for read\n",
		       parms->img_in_2_fn);
    }

    if (!PlmImage::compare_headers (img1, img2)) {
	print_and_exit ("Error: image sizes do not match\n");
    }

    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;
    FloatImageType::Pointer fi1 = img1->itk_float ();
    FloatImageType::Pointer fi2 = img2->itk_float ();

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

    FloatImageType::RegionType rg1 = fi1->GetLargestPossibleRegion ();

#if defined (commentout)
    

    FloatIteratorType it (img, rg);

    int first = 1;
    float min_val, max_val;
    int num = 0;
    double sum = 0.0;

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	float v = it.Get();
	if (first) {
	    min_val = max_val = v;
	}
	if (min_val > v) min_val = v;
	if (max_val < v) max_val = v;
	sum += v;
	num ++;
    }

    printf ("MIN %f AVE %f MAX %f NUM %d\n",
	    min_val, max_val, (float) (sum / num), num);
#endif
}

static void
compare_print_usage (void)
{
    printf ("Usage: plastimatch compare image-1 image-2\n"
	    );
    exit (-1);
}

static void
compare_parse_args (Compare_parms* parms, int argc, char* argv[])
{
    if (argc != 4) {
	compare_print_usage ();
    }
    
    strncpy (parms->img_in_1_fn, argv[2], _MAX_PATH);
    strncpy (parms->img_in_2_fn, argv[3], _MAX_PATH);
}

void
do_command_compare (int argc, char *argv[])
{
    Compare_parms parms;
    
    compare_parse_args (&parms, argc, argv);

    compare_main (&parms);
}
