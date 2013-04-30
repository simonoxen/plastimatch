/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"

#include "itk_adjust.h"
#include "itk_image_save.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "plm_math.h"
#include "pcmd_adjust.h"
#include "print_and_exit.h"

static void
do_itk_adjust (FloatImageType::Pointer image, Adjust_parms *parms)
{
    Adjustment_list al;
    const char* c = parms->pw_linear.c_str();
    bool have_curve = false;

    while (1) {
        int n;
        float f1, f2;
        int rc = sscanf (c, " %f , %f %n", &f1, &f2, &n);
        if (rc < 2) {
            break;
        }
        have_curve = true;

        /* Look for end-caps */
        if (!is_number(f1)) {
            if (al.size() == 0) {
                f1 = -std::numeric_limits<float>::max();
            } else {
                f1 = std::numeric_limits<float>::max();
            }
        }
        /* Append (x,y) pair to list */
        al.push_back (std::make_pair (f1, f2));

        /* Look for next pair in string */
        c += n;
        if (*c == ',') c++;
    }
    
    if (have_curve) {
        itk_adjust (image, al);
    } else {
        print_and_exit ("Error parsing --pw-linear option: %s\n",
            parms->pw_linear.c_str());
    }
}

static void
adjust_main (Adjust_parms* parms)
{
    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;

    Plm_image *plm_image = plm_image_load (
	(const char*) parms->img_in_fn, 
	PLM_IMG_TYPE_ITK_FLOAT);
    FloatImageType::Pointer img = plm_image->m_itk_float;
    FloatImageType::RegionType rg = img->GetLargestPossibleRegion ();
    FloatIteratorType it (img, rg);

    if (parms->have_truncate_above) {
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    if (v > parms->truncate_above) {
		it.Set (parms->truncate_above);
	    }
	}
    }

    if (parms->have_truncate_below) {
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    if (v < parms->truncate_below) {
		it.Set (parms->truncate_below);
	    }
	}
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

    if (parms->have_scale) {
	it.GoToBegin();
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    v = v * parms->scale;
	    it.Set (v);
	}
    }

    if (parms->have_stretch) {
	float vmin, vmax;
	it.GoToBegin();
	vmin = it.Get();
	vmax = it.Get();
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    if (v > vmax) {
		vmax = v;
	    } else if (v < vmin) {
		vmin = v;
	    }
	}
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float v = it.Get();
	    v = (v - vmin) / (vmax - vmin);
	    v = (v + parms->stretch[0]) 
		* (parms->stretch[1] - parms->stretch[0]);
	    it.Set (v);
	}
    }

    if (parms->pw_linear.not_empty()) {
        do_itk_adjust (img, parms);
    }

    if (parms->output_dicom) {
	itk_image_save_short_dicom (
	    img, (const char*) parms->img_out_fn, 0);
    } else {
	if (parms->output_type) {
	    plm_image->convert (parms->output_type);
	}
	plm_image->save_image ((const char*) parms->img_out_fn);
    }

    delete plm_image;
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

    /* Piecewise linear adjustment string */
    if (parser->option ("pw-linear")) {
        parms->pw_linear = parser->get_string("pw-linear").c_str();
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
