/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"

#include "itk_adjust.h"
#include "itk_histogram_matching.h"
#include "itk_image_save.h"
#include "itk_image_shift_scale.h"
#include "itk_image_stats.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "plm_math.h"
#include "pcmd_adjust.h"
#include "print_and_exit.h"


static void
adjust_main (Adjust_parms* parms)
{
    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;

    Plm_image::Pointer plm_image = plm_image_load (
	parms->img_in_fn, 
	PLM_IMG_TYPE_ITK_FLOAT);
    FloatImageType::Pointer img = plm_image->m_itk_float;
    FloatImageType::RegionType rg = img->GetLargestPossibleRegion ();
    FloatIteratorType it (img, rg);

    /* Read Reference image if available */
    FloatImageType::Pointer ref_img;
    if (parms->img_ref_fn != "") {
        Plm_image::Pointer plm_ref_image = plm_image_load(
            parms->img_ref_fn,
            PLM_IMG_TYPE_ITK_FLOAT);
        ref_img = plm_ref_image->m_itk_float;
    }

    /* Read mask images if available */
    UCharImageType::Pointer mask_in, mask_ref;
    if (parms->mask_in_fn != "") {
        Plm_image::Pointer plm_mask_image = plm_image_load(
            parms->mask_in_fn, PLM_IMG_TYPE_ITK_UCHAR);
        mask_in = plm_mask_image->m_itk_uchar;
    }
    if (parms->mask_ref_fn != "") {
        Plm_image::Pointer plm_refmask_image = plm_image_load(
            parms->mask_ref_fn, PLM_IMG_TYPE_ITK_UCHAR);
        mask_ref = plm_refmask_image->m_itk_uchar;
    }

    if (parms->have_ab_scale) {
	it.GoToBegin();
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    float d_per_fx = v / parms->num_fx;
	    v = v * (parms->alpha_beta + d_per_fx) 
		/ (parms->alpha_beta + parms->norm_dose_per_fx);
	    it.Set (v);
	}
    }

    if (parms->pw_linear != "") {
        img = itk_adjust (img, parms->pw_linear);
        plm_image->set_itk (img);
    }

    if (parms->do_hist_match) {
        img = itk_histogram_matching(img, ref_img, parms->hist_th,
            parms->hist_levels, parms->hist_points);
        plm_image->set_itk(img);
    }

    if (parms->do_linear_match) {
        double min, max, avg_in, sigma_in, avg_ref, sigma_ref;
        int num, nonzero;
        if (parms->mask_in_fn != "" && parms->mask_ref_fn != "") {
            itk_masked_image_stats(img, mask_in, STATS_OPERATION_INSIDE,
                &min, &max, &avg_in, &nonzero, &num, &sigma_in);
            itk_masked_image_stats(ref_img, mask_ref, STATS_OPERATION_INSIDE,
                &min, &max, &avg_ref, &nonzero, &num, &sigma_ref);
        } else {
            itk_image_stats(img, &min, &max, &avg_in, &nonzero, &num, &sigma_in);
            itk_image_stats(ref_img, &min, &max, &avg_ref, &nonzero, &num, &sigma_ref);
        }
        float shift, scale;
        scale = float(sigma_ref / sigma_in);
        shift = float(avg_ref - avg_in * scale);
        itk_image_shift_scale(img, shift, scale);
    }
    if (parms->do_linear) {
        itk_image_shift_scale(img, parms->shift, parms->scale);
    }

    if (parms->output_dicom) {
        plm_image->save_short_dicom (parms->img_out_fn.c_str(), 0);
    } else {
        if (parms->output_type) {
            plm_image->convert (parms->output_type);
        }
        plm_image->save_image (parms->img_out_fn);
    }
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options]\n", argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Adjust_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Input files */
    parser->add_long_option ("", "input", 
        "input directory or filename", 1, "");

    /* Output files */
    parser->add_long_option ("", "output", "output image", 1, "");

    /* Adjustment string */
    parser->add_long_option ("", "pw-linear", 
        "a string that forms a piecewise linear map from "
        "input values to output values, of the form "
        "\"in1,out1,in2,out2,...\"", 
        1, "");

    /* histogram matching options */
    parser->add_long_option ("", "hist-match",
        "reference image for histogram matching. You must\n"
        "specify --hist-levels and --hist-points",
        1, "");
    parser->add_long_option ("", "hist-levels",
        "number of histogram bins for histogram matching",
        1, "");
    parser->add_long_option("", "hist-points",
        "number of match points for histogram matching",
        1, "");
    parser->add_long_option("", "hist-threshold",
        "threshold at mean intensity (simple background exclusion", 0);

    /* linear matching options */
    parser->add_long_option("", "linear-match",
        "reference image for linear matching with mean and std",
        1, "");
    parser->add_long_option("", "ref-mask",
        "reference image mask for statistics calculations (linear-match)",
        1, "");
    parser->add_long_option("", "input-mask",
        "input image mask for statistics calculations (linear-match)",
        1, "");

    parser->add_long_option("", "linear",
        "shift and scale image intensities", 0);
    parser->add_long_option("", "shift",
        "shift value for linear adjustment (default 0.0)", 1, "");
    parser->add_long_option("", "scale",
        "scale value for linear adjustment (default 1.0)", 1, "");


    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an output file was given */
    if (!parser->option ("input")) {
	throw (dlib::error ("Error.  Please specify an input file "
		"using the --input option"));
    }

    /* Check that an output file was given */
    if (!parser->option ("output")) {
	throw (dlib::error ("Error.  Please specify an output file "
		"using the --output option"));
    }

    /* Copy input filenames to parms struct */
    parms->img_in_fn = parser->get_string("input").c_str();

    /* Output files */
    parms->img_out_fn = parser->get_string("output").c_str();

    /* Check if only one method is used */
    if (!(parser->option("pw-linear") ^ parser->option("hist-match") ^ parser->option("linear-match")
            ^ parser->option("linear"))) {
        throw (dlib::error("Error. Please use only one of --pw-linear, --hist-match"
                           "--linear-match or --linear"));
    }

    /* Piecewise linear adjustment string */
    if (parser->option ("pw-linear")) {
        parms->pw_linear = parser->get_string("pw-linear").c_str();
    }

    if (parser->option ("hist-match")) {
        if (!parser->option("hist-levels") || !parser->option("hist-points")) {
            throw (dlib::error ("Error. Please specify number of bins and match points"
                                "\nusing the --hist-levels and --hist-points options"));
        }
        parms->do_hist_match = true;
        parms->img_ref_fn = parser->get_string("hist-match");
        parms->hist_levels = parser->get_int("hist-levels");
        parms->hist_points = parser->get_int("hist-points");
        parms->hist_th = bool(parser->option("hist-threshold"));
    }
    if (parser->option ("linear-match")) {
        parms->do_linear_match = true;
        parms->img_ref_fn = parser->get_string("linear-match");
        parms->mask_ref_fn = parser->get_string("ref-mask");
        parms->mask_in_fn = parser->get_string("input-mask");
    }
    if (parser->option ("linear")) {
        parms->do_linear = true;
        if (parser->option ("shift")) {
            parms->shift = parser->get_float("shift");
        }
        if (parser->option ("scale")) {
            parms->scale = parser->get_float("scale");
        }
    }
}

void
do_command_adjust (int argc, char *argv[])
{
    Adjust_parms parms;
    
    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    adjust_main (&parms);
}
