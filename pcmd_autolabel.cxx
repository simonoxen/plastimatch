/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <iostream>
#include "itkImageRegionIterator.h"
#include "dlib/data_io.h"
#include "dlib/svm.h"

#include "autolabel_ransac_est.h"
#include "bstring_util.h"
#include "itk_image.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_image_patient_position.h"
#include "print_and_exit.h"
#include "thumbnail.h"

class Autolabel_parms {
public:
    CBString input_fn;
    CBString output_fn;
    CBString network_fn;
};

/* ITK typedefs */
typedef itk::ImageRegionConstIterator< FloatImageType > FloatIteratorType;

/* Dlib typedefs */
typedef std::map < unsigned long, double > sparse_sample_type;
typedef dlib::matrix < 
    sparse_sample_type::value_type::second_type, 256, 1
    > dense_sample_type;
typedef dlib::radial_basis_kernel < dense_sample_type > kernel_type;

void
do_autolabel (Autolabel_parms *parms)
{
    FILE *fp;

    /* Load network */
    dlib::decision_function<kernel_type> dlib_network;
    std::ifstream fin ((const char*) parms->network_fn, std::ios::binary);
    printf ("Trying to deserialize...\n");
    deserialize (dlib_network, fin);
    printf ("Done.\n");

    /* Load input image */
    Plm_image pli ((const char*) parms->input_fn, PLM_IMG_TYPE_ITK_FLOAT);

    Thumbnail thumbnail;
    thumbnail.set_input_image (&pli);
    thumbnail.set_thumbnail_dim (16);
    thumbnail.set_thumbnail_spacing (25.0f);

    /* Open output file (txt format) */
    fp = fopen ((const char*) parms->output_fn, "w");
    if (!fp) {
	print_and_exit ("Failure to open file for write: %s\n", 
	    (const char*) parms->output_fn);
    }

    /* Create a vector to hold the results */
    Autolabel_point_vector apv;

    /* Loop through slices, and predict location for each slice */
    Plm_image_header pih (&pli);
    for (int i = 0; i < pih.Size(2); i++) {

	/* Create slice thumbnail */
	float loc = pih.m_origin[2] + i * pih.m_spacing[2];
	thumbnail.set_slice_loc (loc);
	FloatImageType::Pointer thumb_img = thumbnail.make_thumbnail ();

	/* Convert to dlib sample type */
	dense_sample_type d;
	FloatIteratorType it (thumb_img, thumb_img->GetLargestPossibleRegion());
	for (int j = 0; j < 256; j++) {
	    d(j) = it.Get();
	    ++it;
	}

	/* Predict the value */
	Autolabel_point ap;
	ap[0] = loc;
	ap[1] = dlib_network (d);
	ap[2] = 0.;
	apv.push_back (ap);
    }

    /* Run RANSAC to refine the estimate */
    //autolabel_ransac_est (apv);

    /* Save the output to a file */
    Autolabel_point_vector::iterator it;
    for (it = apv.begin(); it != apv.end(); it++) {
	fprintf (fp, "%g %g %g\n", (*it)[0], (*it)[1], (*it)[2]);
    }
    
    fclose (fp);
}


static void
usage_fn (dlib::Plm_clp* parser)
{
    std::cout << "Usage: plastimatch autolabel [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}


static void
parse_fn (
    Autolabel_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    parser->add_long_option ("h", "help", "Display this help message");

    /* Basic options */
    parser->add_long_option ("", "output", 
	"Output csv filename (required)", 1, "");
    parser->add_long_option ("", "input", 
	"Input image filename (required)", 1, "");
    parser->add_long_option ("", "network", 
	"Input trained network filename (required)", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Check if the -h option was given */
    parser->check_help ();

    /* Check that an input file was given */
    parser->check_required ("input");

    /* Check that an output file was given */
    parser->check_required ("output");

    /* Check that an network file was given */
    parser->check_required ("network");

    /* Copy values into output struct */
    parms->output_fn = parser->get_string("output").c_str();
    parms->input_fn = parser->get_string("input").c_str();
    parms->network_fn = parser->get_string("network").c_str();
}

void
do_command_autolabel (int argc, char *argv[])
{
    Autolabel_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_autolabel (&parms);
}
