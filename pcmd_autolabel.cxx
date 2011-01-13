/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "getopt.h"
#include "dlib/data_io.h"
#include "dlib/svm.h"

#include "bstring_util.h"
#include "itk_image.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_image_patient_position.h"
#include "thumbnail.h"

typedef struct autolabel_parms Autolabel_parms;
struct autolabel_parms {
    CBString input_fn;
    CBString output_fn;
    CBString network_fn;
};

/* Dlib typedefs */
typedef std::map < unsigned long, double > sparse_sample_type;
typedef dlib::matrix < sparse_sample_type::value_type::second_type,0,1
		 > dense_sample_type;
typedef dlib::radial_basis_kernel < dense_sample_type > kernel_type;

static void
print_usage (void)
{
    printf (
	"Usage: plastimatch autolabel [options]\n"
	"Options:\n"
	"   --input image\n"
	"   --network dlib-network\n"
	"   --output csv-file\n"
    );
    exit (1);
}

void
parse_args (Autolabel_parms *parms, int argc, char *argv[])
{
    int i;

    /* Set default values */
    parms->network_fn = "C:/gcs6/foo.dat";

    for (i = 2; i < argc; i++) {
	if (argv[i][0] != '-') break;
	if (!strcmp (argv[i], "--input")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    parms->input_fn = argv[i];
	}
	else if (!strcmp (argv[i], "--network")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    parms->network_fn = argv[i];
	}
	else if (!strcmp (argv[i], "--output")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    parms->output_fn = argv[i];
	}
	else {
	    print_usage ();
	    break;
	}
    }

    if (bstring_empty (parms->input_fn)) {
	fprintf (stderr, "Error, you must supply an --input option\n");
	exit (-1);
    }
    if (bstring_empty (parms->output_fn)) {
	fprintf (stderr, "Error, you must supply an --output option\n");
	exit (-1);
    }
}

void
autolabel_main (int argc, char *argv[])
{
    Autolabel_parms parms;

    parse_args (&parms, argc, argv);

    /* Load network */
    dlib::decision_function<kernel_type> dlib_network;
    std::ifstream fin ((const char*) parms.network_fn, std::ios::binary);
    deserialize (dlib_network, fin);

    /* Load input image */
    Plm_image pli ((const char*) parms.input_fn, PLM_IMG_TYPE_ITK_FLOAT);

    Thumbnail thumbnail;
    thumbnail.set_input_image (&pli);
    thumbnail.set_thumbnail_dim (16);
    thumbnail.set_thumbnail_spacing (25.0f);

    Plm_image_header pih (&pli);
    for (int i = 0; i < pih.Size(2); i++) {
	float loc = pih.m_origin[2] + i * pih.m_spacing[2];
	thumbnail.set_slice_loc (loc);
	FloatImageType::Pointer thumb_img = thumbnail.make_thumbnail ();
    }
}

void
do_command_autolabel (int argc, char *argv[])
{
    if (argc < 2) {
	print_usage ();
    }

    autolabel_main (argc, argv);
}
