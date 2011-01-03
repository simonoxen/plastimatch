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
typedef matrix<sparse_sample_type::value_type::second_type,0,1> dense_sample_type;

static void
parse_args (clp& parser, int argc, char* argv[])
{
    try {
        parser.add_option("a",
	    "Choose the learning algorithm: {krls,krr,mlp,svr}.",1);
        parser.add_option("h","Display this help message.");
        parser.add_option("help","Display this help message.");
        parser.add_option("in","A libsvm-formatted file to test.",1);
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

static void
krls_test (
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    typedef radial_basis_kernel<dense_sample_type> kernel_type;

    // Here we declare an instance of the krls object.  The first argument 
    // to the constructor is the kernel we wish to use.  The second is 
    // a parameter that determines the numerical accuracy with which 
    // the object will perform part of the regression algorithm.  
    // Generally smaller values give better results but cause the 
    // algorithm to run slower.  You just have to play with it to 
    // decide what balance of speed and accuracy is right for your problem.
    // Here we have set it to 0.001.
    krls<kernel_type> net(kernel_type(0.01),0.001);

    // Split into training set and testing set
    float training_pct = 0.8;
    randomize_samples(dense_samples, labels);
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
krr_test (
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    typedef radial_basis_kernel<dense_sample_type> kernel_type;
    krr_trainer<kernel_type> trainer;

    // Here we set the kernel we want to use for training.   The 
    // radial_basis_kernel has a parameter called gamma that we need to 
    // determine.  As a rule of thumb, a good gamma to try is 
    // 1.0/(mean squared distance between your sample points).  So 
    // below we are using a similar value computed from at most 2000
    //  randomly selected samples.
    const double gamma = 3.0 / 
	compute_mean_squared_distance(randomly_subsample(dense_samples, 2000));
    std::cout << "using gamma of " << gamma << std::endl;
    trainer.set_kernel(kernel_type(gamma));

    // The krr_trainer has the ability to tell us the leave-one-out 
    // cross-validation accuracy.  The train() function has an optional 
    // 3rd argument and if we give it a double it will give us back 
    // the LOO error.
    double loo_error;
    trainer.train(dense_samples, labels, loo_error);
    std::cout << "mean squared LOO error: " << loo_error << std::endl;
}

static void
mlp_test (
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    // Create a multi-layer perceptron network.
    const int num_input = dense_samples[0].size();
    const int num_hidden = 5;
    printf ("Creating ANN with size (%d, %d)\n", num_input, num_hidden);
    mlp::kernel_1a_c net(num_input,num_hidden);

    // Dlib barfs if output values are not normalized to [0,1]
    double label_min = *(std::min_element (labels.begin(), labels.end()));
    double label_max = *(std::max_element (labels.begin(), labels.end()));
    std::vector<double>::iterator it;
    for (it = labels.begin(); it != labels.end(); it++) {
	(*it) = ((*it) - label_min) / (label_max - label_min);
    }

    // Split into training set and testing set
    float training_pct = 0.8;
    randomize_samples(dense_samples, labels);
    unsigned int training_samples = (unsigned int) floor (
	training_pct * dense_samples.size());

    // Dlib doesn't seem to come with any batch training functions for mlp.
    // Also, note that only backprop is supported.
    const int num_iterations = 5000;
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
    std::vector<dense_sample_type>& dense_samples,
    std::vector<double>& labels
)
{
    // Now we are making a typedef for the kind of kernel we want to use.  
    // I picked the radial basis kernel because it only has one parameter 
    // and generally gives good results without much fiddling.
    typedef radial_basis_kernel<dense_sample_type> kernel_type;

    // Now setup a SVR trainer object.  It has three parameters, the kernel and
    // two parameters specific to SVR.  
    svr_trainer<kernel_type> trainer;
    trainer.set_kernel(kernel_type(0.01));

    // This parameter is the usual regularization parameter.  It determines
    // the trade-off between trying to reduce the training error or 
    // allowing more errors but hopefully improving the generalization 
    // of the resulting function.  Larger values encourage exact fitting 
    // while smaller values of C may encourage better generalization.
    //trainer.set_c(10);
    trainer.set_c(1000);

    // Epsilon-insensitive regression means we do regression but stop 
    // trying to fit a data point once it is "close enough" to its 
    // target value.  This parameter is the value that controls what 
    // we mean by "close enough".  In this case, I'm saying I'm happy 
    // if the resulting regression function gets within 0.001 of the 
    // target value.
    trainer.set_epsilon_insensitivity(0.001);

    // We can also do 10-fold cross-validation and find the mean squared 
    // error.  Note that we need to randomly shuffle the samples first.
    // See the svm_ex.cpp for a discussion of why this is important.
    randomize_samples(dense_samples, labels);
    std::cout 
	<< "MSE (10-fold): "
	<< cross_validate_regression_trainer(
	    trainer, dense_samples, labels, 10) 
	<< std::endl;
}

int 
main(int argc, char* argv[])
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

    if (!option_alg) {
	// Do KRR if user didn't specify an algorithm
	std::cout << "No algorithm specified, default to KRR\n";
	krr_test (dense_samples, labels);
    }
    else if (!strcmp (option_alg.argument().c_str(), "krls")) {
	krls_test (dense_samples, labels);
    }
    else if (!strcmp (option_alg.argument().c_str(), "krr")) {
	krr_test (dense_samples, labels);
    }
    else if (!strcmp (option_alg.argument().c_str(), "mlp")) {
	mlp_test (dense_samples, labels);
    }
    else if (!strcmp (option_alg.argument().c_str(), "svr")) {
	svr_test (dense_samples, labels);
    }
    else {
	fprintf (stderr, 
	    "Error, algorithm \"%s\" is unknown.\n"
	    "Please use -h to see the command line options\n",
	    option_alg.argument().c_str());
	exit (-1);
    }
}
