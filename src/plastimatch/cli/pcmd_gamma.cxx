/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "gamma_dose_comparison.h"
#include "logfile.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "plm_math.h"
#include "pcmd_gamma.h"
#include "print_and_exit.h"

static void
gamma_main (Gamma_parms* parms)
{
    Gamma_dose_comparison gdc;

    gdc.set_reference_image (parms->ref_image_fn.c_str());
    gdc.set_compare_image (parms->cmp_image_fn.c_str());	

    gdc.set_spatial_tolerance (parms->dta_tolerance);
	gdc.set_dose_difference_tolerance (parms->dose_tolerance);
	if (parms->have_reference_dose) {
        gdc.set_reference_dose (parms->reference_dose);
    }
    gdc.set_gamma_max (parms->gamma_max);

	/*Extended by YK*/
	gdc.set_interp_search(parms->b_interp_search);//default: false
	gdc.set_local_gamma(parms->b_local_gamma);//default: false
	gdc.set_compute_full_region(parms->b_compute_full_region);//default: false
	gdc.set_resample_nn(parms->b_resample_nn); //default: false

	if (parms->f_inherent_resample_mm > 0.0){
		gdc.set_inherent_resample_mm(parms->f_inherent_resample_mm);
	}

	if (parms->f_analysis_threshold > 0){
		gdc.set_analysis_threshold(parms->f_analysis_threshold);//0.1 = 10%
	}

    gdc.run ();

    if (parms->out_image_fn != "") {
        Plm_image::Pointer gamma_image = gdc.get_gamma_image ();
        gamma_image->save_image (parms->out_image_fn);

		//YKdebug
		//gdc.get_fail_image()->save_image("<fail_image_map_path>");
		//YKdebug
    }

	if (parms->out_report_fn != "") {
		//Export output text using //gdc.get_report_string();
		std::ofstream fout;
		fout.open(parms->out_report_fn.c_str());
		if (!fout.fail()){
			fout << "Reference_file_name\t" << parms->ref_image_fn.c_str() << std::endl;
			fout << "Compare_file_name\t" << parms->cmp_image_fn.c_str() << std::endl;

			fout << gdc.get_report_string();
			fout.close();
		}		
	}

    lprintf ("Pass rate = %2.6f %%\n", gdc.get_pass_fraction() * 100.0);
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options] image_1 image_2\n", argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Gamma_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Output files */
    parser->add_long_option ("", "output", "output image", 1, "");

    /* Gamma options */
    parser->add_long_option ("", "dose-tolerance", 
        "The scaling coefficient for dose difference in percent "
        "(default is .03)", 1, ".03");
    parser->add_long_option ("", "dta-tolerance", 
        "The distance-to-agreement (DTA) scaling coefficient in mm "
        "(default is 3)", 1, "3");
    parser->add_long_option ("", "reference-dose", 
        "The prescription dose used to compute dose tolerance; if not "
        "specified, then maximum dose in reference volume is used",
        1, "");
    parser->add_long_option ("", "gamma-max", 
        "The maximum value of gamma to compute; smaller values run faster "
        "(default is 2.0)", 1, "2.0");


	/* extended by YK*/
	parser->add_long_option("", "interp-search", "With this option, smart interpolation search will be applied in points near the reference point. This will eliminate the need of fine resampling using inherent-resampling option. However, it will take longer time. ", 0);

	parser->add_long_option("", "local-gamma", "With this option, dose difference (e.g. 3%) is calculated based on local dose difference. Otherwise, reference dose will be used. ",0);

	parser->add_long_option("", "compute-full-region", "With this option, gamma values will be calculated over the entire image. Otherwise, gamma = 0.0 will be assigned for below-threshold regions with speeding up the calculation", 0);

	parser->add_long_option("", "resample-nn", "With this option, nearest neighbor will be used instead of linear interpolation in resampling comp. image wrt ref. image as well as in inherent-resampling", 0);

	parser->add_long_option("", "inherent-resample",
		"Spacing value in [mm]. Alternative to make the mask. based on the specified value here, both ref and comp image will be resampled. if < 0, this option is disabled.  "
		"(default is -1.0)", 1, "-1.0");

	parser->add_long_option("", "analysis-threshold",
		"Analysis threshold for dose in float (default: 0.1 = 10%). This will be used in conjunction with the reference dose value or maximum dose if not specified"
		"(default is 0.1)", 1, "0.1");	

	parser->add_long_option("", "output-text", "Text file path for gamma evaluation result", 1, "");	

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that two input files were given */
    if (parser->number_of_arguments() < 2) {
	throw (dlib::error ("Error.  You must specify two input files"));
	
    } else if (parser->number_of_arguments() > 2) {
	std::string extra_arg = (*parser)[1];
	throw (dlib::error ("Error.  Extra argument " + extra_arg));
    }

    /* Input files */
    parms->ref_image_fn = (*parser)[0].c_str();
    parms->cmp_image_fn = (*parser)[1].c_str();

    /* Output files */
    if (parser->option ("output")) {
        parms->out_image_fn = parser->get_string("output").c_str();
    }


    /* Gamma options */
    parms->dose_tolerance = parser->get_float("dose-tolerance");
    parms->dta_tolerance = parser->get_float("dta-tolerance");
    parms->gamma_max = parser->get_float("gamma-max");
    if (parser->option("reference-dose")) {
        parms->have_reference_dose = true;
        parms->reference_dose = parser->get_float("reference-dose");
    }

	if (parser->option("interp-search")) {
		parms->b_interp_search = true;
	}
	
	if (parser->option("local-gamma")) {
		parms->b_local_gamma = true;
	}

	if (parser->option("compute-full-region")) {		
			parms->b_compute_full_region = true;
	}

	if (parser->option("resample-nn")) {
		parms->b_resample_nn = true;
	}		

	if (parser->option("inherent-resample")) {
		parms->f_inherent_resample_mm = parser->get_float("inherent-resample");
	}

	if (parser->option("analysis-threshold")) {
		parms->f_analysis_threshold = parser->get_float("analysis-threshold");
	}

	/* Output file for text report */
	if (parser->option("output-text")) {
		parms->out_report_fn = parser->get_string("output-text").c_str();
	}
}

void
do_command_gamma (int argc, char *argv[])
{
    Gamma_parms parms;

    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    gamma_main (&parms);
}
