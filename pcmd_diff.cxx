/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkSubtractImageFilter.h"
#include "itkImageRegionIterator.h"
#include "getopt.h"
#include "pcmd_diff.h"
#include "plm_image.h"
#include "itk_image.h"

static void
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

    if (!Plm_image::compare_headers (img1, img2)) {
	print_and_exit ("Error: image sizes do not match\n");
    }

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
    FloatImageType::Pointer diff = sub_filter->GetOutput ();

    itk_image_save_float (diff, (const char*) parms->img_out_fn);
}

static void
diff_print_usage (void)
{
    printf ("Usage: plastimatch diff image_in_1 image_in_2 image_out\n"
	    );
    exit (-1);
}

static void
diff_parse_args (Diff_parms* parms, int argc, char* argv[])
{
    if (argc != 5) {
	diff_print_usage ();
    }
    
    parms->img_in_1_fn = argv[2];
    parms->img_in_2_fn = argv[3];
    parms->img_out_fn = argv[4];
}

void
do_command_diff (int argc, char *argv[])
{
    Diff_parms parms;
    
    diff_parse_args (&parms, argc, argv);

    diff_main (&parms);
}
