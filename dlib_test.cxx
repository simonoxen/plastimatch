// The contents of this file are in the public domain. 
// See LICENSE_FOR_EXAMPLE_PROGRAMS.txt (in dlib)
/*
    This is a command line program that can try different regression 
    algorithms on a libsvm-formatted data set.
*/
#include <iostream>
#include <map>
#include <vector>

#include "dlib/cmd_line_parser.h"
#include "dlib/data_io.h"
#include "dlib/mlp.h"
#include "dlib/svm.h"

using namespace dlib;

typedef dlib::cmd_line_parser<char>::check_1a_c clp;
typedef std::map<unsigned long, double> sparse_sample_type;
typedef matrix< sparse_sample_type::value_type::second_type,0,1
		> dense_sample_type;

static void
parse_args (clp& parser, int argc, char* argv[])
{
    try {
	// Algorithm-independent options
        parser.add_option ("a",
	    "Choose the learning algorithm: {krls,krr,mlp,svr}.",1);
        parser.add_option ("h","Display this help message.");
        parser.add_option ("help","Display this help message.");
        parser.add_option ("k",
	    "Learning kernel (for krls,krr,svr methods): {lin,rbk}.",1);
        parser.add_option ("in","A libsvm-formatted file to test.",1);
        parser.add_option ("normalize",
	    "Normalize the sample inputs to zero-mean unit variance?");

	// Algorithm-specific options
        parser.add_option ("rbk-gamma",
	    "Width of radial basis kernels: {float}.",1);
        parser.add_option ("krls-tolerance",
	    "Numerical tolerance of krls linear dependency test: {float}.",1);
        parser.add_option ("mlp-hidden-units",
	    "Number of hidden units in mlp: {integer}.",1);
        parser.add_option ("mlp-num-iterations",
	    "Number of epochs to train the mlp: {integer}.",1);
        parser.add_option ("svr-c",
	    "SVR regularization parameter \"C\": "
	    "{float}.",1);
        parser.add_option ("svr-epsilon-insensitivity",
	    "SVR fitting tolerance parameter: "
	    "{float}.",1);

	// Parse the command line arguments
        parser.parse(argc,argv);

	// Check that options aren't given multiple times
        const char* one_time_opts[] = {"a", "h", "help", "in"};
        parser.check_one_time_options(one_time_opts);

        // Check if the -h option was given
        if (parser.option("h") || parser.option("help")) {
	    std::cout << "Usage: dlib_test [-a algorithm] --in input_file\n";
            parser.print_options(std::cout);
	    std::cout << std::endl;
	    exit (0);
        }

	// Check that an input file was given
        if (!parser.option("in")) {
	    std::cout 
		<< "Error in command line:\n"
		<< "You must specify an input file with the --in option.\n"
		<< "\nTry the -h option for more information\n";
	    exit (0);
	}
    }
    catch (std::exception& e) {
        // Catch cmd_line_parse_error exceptions and print usage message.
	std::cout << e.what() << std::endl;
	exit (1);
    }
    catch (...) {
	std::cout << "Some error occurred" << std::endl;
    }
}

static const char*
get_kernel (
    clp& parser
)
{
    const char* kernel = "rbk";
    if (parser.option ("k")) {
	kernel = parser.option("k").argument().c_str();
    }
    return kernel;
}

static double
get_rbk_gamma (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples
)
{
    // The radial_basis_kernel has a parameter called gamma that we 
    // need to determine.  As a rule of thumb, a good gamma to try is 
    // 1.0/(mean squared distance between your sample points).  So 
    // below we are using a similar value computed from at most 2000
    //  randomly selected samples.
    double gamma;
    if (parser.option ("rbk-gamma")) {
	gamma = sa = parser.option("rbk-gamma").argument();
    } else {
	gamma = 3.0 / compute_mean_squared_distance (
	    randomly_subsample (dense_samples, 2000));
    }
    return gamma;
}

static double
get_krls_tolerance (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples
)
{
    // Here we declare an instance of the krls object.  The first argument 
    // to the constructor is the kernel we wish to use.  The second is 
    // a parameter that determines the numerical accuracy with which 
    // the object will perform part of the regression algorithm.  
    // Generally smaller values give better results but cause the 
    // algorithm to run slower.  You just have to play with it to 
    // decide what balance of speed and accuracy is right for your problem.
    // Here we have set it to 0.001.
    double krls_tolerance = 0.001;
    if (parser.option ("krls-tolerance")) {
	krls_tolerance = sa = parser.option("krls-tolerance").argument();
    }
    return krls_tolerance;
}

static double
get_mlp_hidden_units (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples
)
{
    int num_hidden = 5;
    if (parser.option ("mlp-hidden-units")) {
	num_hidden = sa = parser.option("mlp-hidden-units").argument();
    }
    return num_hidden;
}

static double
get_mlp_num_iterations (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples
)
{
    int num_iterations = 5000;
    if (parser.option ("mlp-num-iterations")) {
	num_iterations = sa = parser.option("mlp-num-iterations").argument();
    }
    return num_iterations;
}

static double
get_svr_c (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples
)
{
    // This parameter is the usual regularization parameter.  It determines
    // the trade-off between trying to reduce the training error or 
    // allowing more errors but hopefully improving the generalization 
    // of the resulting function.  Larger values encourage exact fitting 
    // while smaller values of C may encourage better generalization.
    //double c = 10.;
    double c = 1000.;
    if (parser.option ("svr-c")) {
	c = sa = parser.option("svr-c").argument();
    }
    return c;
}

static double
get_svr_epsilon_insensitivity (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples
)
{
    // Epsilon-insensitive regression means we do regression but stop 
    // trying to fit a data point once it is "close enough" to its 
    // target value.  This parameter is the value that controls what 
    // we mean by "close enough".  In this case, I'm saying I'm happy 
    // if the resulting regression function gets within 0.001 of the 
    // target value.
    double epsilon_insensitivity = 0.001;
    if (parser.option ("svr-epsilon-insensitivity")) {
	epsilon_insensitivity 
	    = sa = parser.option("svr-epsilon-insensitivity").argument();
    }
    return epsilon_insensitivity;
}

static void
krls_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    typedef radial_basis_kernel<dense_sample_type> kernel_type;

    double gamma = get_rbk_gamma (parser, dense_samples);
    double krls_tolerance = get_krls_tolerance (parser, dense_samples);
    krls<kernel_type> net (kernel_type(gamma), krls_tolerance);

    // Split into training set and testing set
    float training_pct = 0.8;
    unsigned int training_samples = (unsigned int) floor (
	training_pct * dense_samples.size());

    // Krls doesn't seem to come with any batch training function
    for (unsigned int j = 0; j < training_samples; j++) {
	net.train (dense_samples[j], labels[j]);
    }

    // Test the performance (sorry, no cross-validation)
    double total_err = 0.0;
    for (unsigned int j = training_samples + 1; j < dense_samples.size(); j++)
    {
	double diff = net(dense_samples[j]) - labels[j];
	total_err += diff * diff;
    }
    std::cout 
	<< "MSE (no cross-validation): " 
	<< total_err / (dense_samples.size() - training_samples) << std::endl;
}

static void
krr_rbk_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    typedef radial_basis_kernel<dense_sample_type> kernel_type;
    krr_trainer<kernel_type> trainer;

    double gamma = get_rbk_gamma (parser, dense_samples);
    trainer.set_kernel(kernel_type(gamma));

    // LOO cross validation
    double loo_error;
    trainer.train(dense_samples, labels, loo_error);
    std::cout << "mean squared LOO error: " << loo_error << std::endl;
}

static void
krr_lin_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    typedef linear_kernel<dense_sample_type> kernel_type;
    krr_trainer<kernel_type> trainer;

    // LOO cross validation
    double loo_error;
    trainer.train(dense_samples, labels, loo_error);
    std::cout << "mean squared LOO error: " << loo_error << std::endl;
}

static void
krr_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    const char* kernel = get_kernel (parser);

    if (!strcmp (kernel, "lin")) {
	krr_lin_test (parser, dense_samples, labels);
    } else if (!strcmp (kernel, "rbk")) {
	krr_rbk_test (parser, dense_samples, labels);
    } else {
	fprintf (stderr, "Unknown kernel type: %s\n", kernel);
	exit (-1);
    }
}

static void
mlp_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    // Create a multi-layer perceptron network.
    const int num_input = dense_samples[0].size();
    int num_hidden = get_mlp_hidden_units (parser, dense_samples);
    printf ("Creating ANN with size (%d, %d)\n", num_input, num_hidden);
    mlp::kernel_1a_c net (num_input, num_hidden);

    // Dlib barfs if output values are not normalized to [0,1]
    double label_min = *(std::min_element (labels.begin(), labels.end()));
    double label_max = *(std::max_element (labels.begin(), labels.end()));
    std::vector<double>::iterator it;
    for (it = labels.begin(); it != labels.end(); it++) {
	(*it) = ((*it) - label_min) / (label_max - label_min);
    }

    // Split into training set and testing set
    float training_pct = 0.8;
    unsigned int training_samples = (unsigned int) floor (
	training_pct * dense_samples.size());

    // Dlib doesn't seem to come with any batch training functions for mlp.
    // Also, note that only backprop is supported.
    int num_iterations = get_mlp_num_iterations (parser, dense_samples);
    for (int i = 0; i < num_iterations; i++) {
	for (unsigned int j = 0; j < training_samples; j++) {
	    net.train (dense_samples[j], labels[j]);
	}
    }

    // Test the performance (sorry, no cross-validation) */
    double total_err = 0.0;
    for (unsigned int j = training_samples + 1; j < dense_samples.size(); j++)
    {
	double diff = net(dense_samples[j]) - labels[j];
	diff = diff * (label_max - label_min);
	total_err += diff * diff;
    }
    std::cout 
	<< "MSE (no cross-validation): " 
	<< total_err / (dense_samples.size() - training_samples) << std::endl;
}

static void
svr_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    typedef radial_basis_kernel<dense_sample_type> kernel_type;
    svr_trainer<kernel_type> trainer;

    double gamma = get_rbk_gamma (parser, dense_samples);
    trainer.set_kernel (kernel_type(gamma));

    double svr_c = get_svr_c (parser, dense_samples);
    trainer.set_c (svr_c);

    double epsilon_insensitivity = get_svr_epsilon_insensitivity (
	parser, dense_samples);
    trainer.set_epsilon_insensitivity (epsilon_insensitivity);

    // 10-fold cross-validation
    std::cout 
	<< "MSE (10-fold): "
	<< cross_validate_regression_trainer(
	    trainer, dense_samples, labels, 10) 
	<< std::endl;
}

int 
main (int argc, char* argv[])
{
    clp parser;

    parse_args(parser, argc, argv);

    const clp::option_type& option_alg = parser.option("a");
    const clp::option_type& option_in = parser.option("in");

    std::vector<sparse_sample_type> sparse_samples;
    std::vector<double> labels;

    load_libsvm_formatted_data (
	option_in.argument(), 
	sparse_samples, 
	labels
    );

    if (sparse_samples.size() < 1) {
	std::cout 
	    << "Sorry, I couldn't find any samples in your data set.\n"
	    << "Aborting the operation.\n";
	exit (0);
    }

    std::vector<dense_sample_type> dense_samples;
    dense_samples = sparse_to_dense (sparse_samples);

    /* GCS FIX: The sparse_to_dense converter adds an extra column, 
       because libsvm files are indexed starting with "1". */
    std::cout 
	<< "Loaded " << sparse_samples.size() << " samples"
	<< std::endl
	<< "Each sample has size " << sparse_samples[0].size() 
	<< std::endl
	<< "Each dense sample has size " << dense_samples[0].size() 
	<< std::endl;

    // Normalize inputs to N(0,1)
    if (parser.option ("normalize")) {
	vector_normalizer<dense_sample_type> normalizer;
	normalizer.train (dense_samples);
	for (unsigned long i = 0; i < dense_samples.size(); ++i) {
	    dense_samples[i] = normalizer (dense_samples[i]);
	}
    }

    // Randomize the order of the samples, labels
    randomize_samples (dense_samples, labels);

    if (!option_alg) {
	// Do KRR if user didn't specify an algorithm
	std::cout << "No algorithm specified, default to KRR\n";
	krr_test (parser, dense_samples, labels);
    }
    else if (!strcmp (option_alg.argument().c_str(), "krls")) {
	krls_test (parser, dense_samples, labels);
    }
    else if (!strcmp (option_alg.argument().c_str(), "krr")) {
	krr_test (parser, dense_samples, labels);
    }
    else if (!strcmp (option_alg.argument().c_str(), "mlp")) {
	mlp_test (parser, dense_samples, labels);
    }
    else if (!strcmp (option_alg.argument().c_str(), "svr")) {
	svr_test (parser, dense_samples, labels);
    }
    else {
	fprintf (stderr, 
	    "Error, algorithm \"%s\" is unknown.\n"
	    "Please use -h to see the command line options\n",
	    option_alg.argument().c_str());
	exit (-1);
    }
}
