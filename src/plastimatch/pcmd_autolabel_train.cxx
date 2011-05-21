/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <iostream>
#include "itkImageRegionIterator.h"
#include "dlib/data_io.h"
#include "dlib/svm.h"

#include "autolabel_trainer.h"
#include "bstring_util.h"
#include "itk_image.h"
#include "pcmd_autolabel_train.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_image_patient_position.h"
#include "print_and_exit.h"
#include "thumbnail.h"

class Autolabel_train_parms {
public:
    CBString input_dir;
    CBString output_libsvm_fn;
    CBString output_network_fn;
    CBString task;
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
do_autolabel_train (Autolabel_train_parms *parms)
{
    Autolabel_trainer at;

    at.load_input_dir ((const char*) parms->input_dir);
    at.set_task ((const char*) parms->task);
    at.save_libsvm ((const char*) parms->output_libsvm_fn);
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch autolabel-train [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Autolabel_train_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    parser->add_default_options ();

    /* Basic options */
    parser->add_long_option ("", "input", 
	"Input directory (required)", 1, "");
    parser->add_long_option ("", "output-libsvm", 
	"Output libsvm training file (required)", 1, "");
    parser->add_long_option ("", "output-network", 
	"Output trained network filename", 1, "");
    parser->add_long_option ("", "task", 
	"Training task (required), choices are "
	"{la,tsv1,tsv2}", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Check if the -h or --version were given */
    parser->check_default_options ();

    /* Check that an input file was given */
    parser->check_required ("input");

    /* Check that a task was given */
    parser->check_required ("output-libsvm");

    /* Check that a task was given */
    parser->check_required ("task");

    /* Copy values into output struct */
    parms->input_dir = parser->get_string("input").c_str();
    parms->output_libsvm_fn = parser->get_string("output-libsvm").c_str();
    parms->output_network_fn = parser->get_string("output-network").c_str();
    parms->task = parser->get_string("task").c_str();
}

void
do_command_autolabel_train (int argc, char *argv[])
{
    Autolabel_train_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_autolabel_train (&parms);
}
