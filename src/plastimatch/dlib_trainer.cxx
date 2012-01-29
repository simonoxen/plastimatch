/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <iostream>
#include <map>
#include <vector>
#include <float.h>
#include <math.h>
#include "dlib/cmd_line_parser.h"
#include "dlib/data_io.h"
#include "dlib/mlp.h"
#include "dlib/revision.h"
#include "dlib/svm.h"

#include "dlib_trainer.h"
#include "math_util.h"

using namespace dlib;

typedef dlib::cmd_line_parser<char>::check_1a_c clp;
typedef std::map<unsigned long, double> sparse_sample_type;
typedef dlib::matrix< sparse_sample_type::value_type::second_type,0,1
		> dense_sample_type;
typedef double label_type;

/* ---------------------------------------------------------------------
   option_range class
   --------------------------------------------------------------------- */
struct option_range {
public:
    bool log_range;
    float min_value;
    float max_value;
    float incr;
public:
    option_range () {
	log_range = false;
	min_value = 0;
	max_value = 100;
	incr = 10;
    }
    void set_option (clp& parser, std::string const& option, 
	float default_val);
    float get_min_value ();
    float get_max_value ();
    float get_next_value (float curr_val);
};

void
option_range::set_option (
    clp& parser,
    std::string const& option, 
    float default_val
)
{
    int rc;

    /* No option specified */
    if (!parser.option (option)) {
	log_range = 0;
	min_value = default_val;
	max_value = default_val;
	incr = 1;
	return;
    }

    /* Range specified */
    rc = sscanf (parser.option(option).argument().c_str(), "%f:%f:%f", 
	&min_value, &incr, &max_value);
    if (rc == 3) {
	log_range = 1;
	return;
    }

    /* Single value specified */
    if (rc == 1) {
	log_range = 0;
	max_value = min_value;
	incr = 1;
	return;
    }

    else {
	std::cerr << "Error parsing option" << option << "\n";
	exit (-1);
    }
}

float 
option_range::get_min_value ()
{
    if (log_range) {
	return exp10_ (min_value);
    } else {
	return min_value;
    }
}

float 
option_range::get_max_value ()
{
    if (log_range) {
	return exp10_ (max_value);
    } else {
	return max_value;
    }
}

float 
option_range::get_next_value (float curr_value)
{
    if (log_range) {
	curr_value = log10 (curr_value);
	curr_value += incr;
	curr_value = exp10_ (curr_value);
    } else {
	curr_value += incr;
    }
    return curr_value;
}

/* ---------------------------------------------------------------------
   global functions
   --------------------------------------------------------------------- */
#if defined (commentout)
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
        parser.add_option ("train-best",
	    "Train and save a network using best parameters", 1);

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
        parser.add_option ("verbose", "Use verbose trainers");

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
#endif

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

static void
get_rbk_gamma (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    option_range& range
) {
    float default_gamma = 3.0 / compute_mean_squared_distance (
	randomly_subsample (dense_samples, 2000));
    range.set_option (parser, "rbk-gamma", default_gamma);
}

static void
get_krls_tolerance (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples, 
    option_range& range
)
{
    float default_krls_tolerance = 0.001;
    range.set_option (parser, "krls-tolerance", default_krls_tolerance);
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

static void
get_svr_c (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples, 
    option_range& range
)
{
    float default_svr_c = 1000.;
    range.set_option (parser, "svr-c", default_svr_c);
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
    std::vector<label_type>& labels
)
{
    typedef radial_basis_kernel<dense_sample_type> kernel_type;
    option_range gamma_range, krls_tol_range;

    get_rbk_gamma (parser, dense_samples, gamma_range);
    get_krls_tolerance (parser, dense_samples, krls_tol_range);

    // Split into training set and testing set
    float training_pct = 0.8;
    unsigned int training_samples = (unsigned int) floor (
	training_pct * dense_samples.size());

    for (float krls_tol = krls_tol_range.get_min_value(); 
	 krls_tol <= krls_tol_range.get_max_value();
	 krls_tol = krls_tol_range.get_next_value (krls_tol))
    {
	for (float gamma = gamma_range.get_min_value(); 
	     gamma <= gamma_range.get_max_value();
	     gamma = gamma_range.get_next_value (gamma))
	{
	    krls<kernel_type> net (kernel_type(gamma), krls_tol);

	    // Krls doesn't seem to come with any batch training function
	    for (unsigned int j = 0; j < training_samples; j++) {
		net.train (dense_samples[j], labels[j]);
	    }

	    // Test the performance (sorry, no cross-validation)
	    double total_err = 0.0;
	    for (unsigned int j = training_samples + 1; 
		 j < dense_samples.size(); j++)
	    {
		double diff = net(dense_samples[j]) - labels[j];
		total_err += diff * diff;
	    }

	    double testset_error = total_err 
		/ (dense_samples.size() - training_samples);
	    printf ("%3.6f %3.6f %3.9f\n", krls_tol, gamma, testset_error);
	}
    }
}

static void
krr_rbk_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<label_type>& labels
)
{
    typedef radial_basis_kernel<dense_sample_type> kernel_type;
    krr_trainer<kernel_type> trainer;
    option_range gamma_range;
    double best_gamma = DBL_MAX;
    float best_loo = FLT_MAX;

    get_rbk_gamma (parser, dense_samples, gamma_range);

    for (float gamma = gamma_range.get_min_value(); 
	 gamma <= gamma_range.get_max_value();
	 gamma = gamma_range.get_next_value (gamma))
    {
	// LOO cross validation
	double loo_error;

	if (parser.option("verbose")) {
	    trainer.set_search_lambdas(logspace(-9, 4, 100));
	    trainer.be_verbose();
	}
	trainer.set_kernel (kernel_type (gamma));

#if DLIB_REVISION == 4093
        /* dlib 17.34 */
	trainer.train (dense_samples, labels, loo_error);
#elif DLIB_MAJOR_VERSION == 17 && DLIB_MINOR_VERSION == 44
        /* dlib 17.44 */
        /* GCS FIX: How to get loo_error from loo_values? */
        std::vector<double> loo_values;
	double lambda_used;
	trainer.train (dense_samples, labels, loo_values, lambda_used);
#else
        error, unknown DLIB version!;
#endif

	if (loo_error < best_loo) {
	    best_loo = loo_error;
	    best_gamma = gamma;
	}
	printf ("10^%f %9.6f\n", log10(gamma), loo_error);
    }

    printf ("Best result: gamma=10^%f (%g), loo_error=%9.6f\n",
	log10(best_gamma), best_gamma, best_loo);
    if (parser.option("train-best")) {
	printf ("Training network with best parameters\n");
	trainer.set_kernel (kernel_type (best_gamma));
	decision_function<kernel_type> best_network = 
	    trainer.train (dense_samples, labels);

	std::ofstream fout (parser.option("train-best").argument().c_str(), 
	    std::ios::binary);
	serialize (best_network, fout);
	fout.close();

#if defined (commentout)
	for (unsigned int j = 0; j < dense_samples.size(); j++) {
	    printf ("%g %g\n", labels[j], best_network(dense_samples[j]));
	}
#endif
    }
}

static void
krr_lin_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<label_type>& labels
)
{
    typedef linear_kernel<dense_sample_type> kernel_type;
    krr_trainer<kernel_type> trainer;

    // LOO cross validation
    double loo_error;
#if DLIB_REVISION == 4093
    /* dlib 17.34 */
    trainer.train(dense_samples, labels, loo_error);
#elif DLIB_MAJOR_VERSION == 17 && DLIB_MINOR_VERSION == 44
    /* dlib 17.44 */
    /* GCS FIX: How to get loo_error from loo_values? */
    std::vector<double> loo_values;
    double lambda_used;
    trainer.train (dense_samples, labels, loo_values, lambda_used);
#else
    error, unknown DLIB version!;
#endif
    std::cout << "mean squared LOO error: " << loo_error << std::endl;
}

static void
krr_test (
    clp& parser,
    std::vector<dense_sample_type>& dense_samples,
    std::vector<label_type>& labels
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
    std::vector<label_type>& labels
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
    std::vector<label_type>::iterator it;
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

/* ---------------------------------------------------------------------
   training functions
   --------------------------------------------------------------------- */
void
Dlib_trainer::save_net (const Pstring& out_net_fn)
{
    std::ofstream fout ((const char*) out_net_fn, std::ios::binary);
    serialize (m_krr_df, fout);
    fout.close();

#if defined (commentout)
	for (unsigned int j = 0; j < dense_samples.size(); j++) {
	    printf ("%g %g\n", labels[j], best_network(dense_samples[j]));
	}
#endif
}

void
Dlib_trainer::save_tsacc (const Pstring& out_net_fn)
{
    FILE *fp = fopen ((const char*) out_net_fn, "w");
    for (unsigned int j = 0; j < m_samples.size(); j++) {
	fprintf (fp, "%g %g\n", m_labels[j], m_krr_df(m_samples[j]));
    }
    fclose (fp);
}

void
Dlib_trainer::set_krr_gamma (
    double gamma_min, double gamma_max, double gamma_step)
{
    m_gamma_search = true;
    m_gamma_min = gamma_min;
    m_gamma_max = gamma_max;
    m_gamma_step = gamma_step;
}

void
Dlib_trainer::train_krr (void)
{
    //typedef radial_basis_kernel<dense_sample_type> kernel_type;
    krr_trainer<Kernel_type> trainer;
    option_range gamma_range;
    double best_gamma = DBL_MAX;
    float best_loo = FLT_MAX;

    //get_rbk_gamma (parser, dense_samples, gamma_range);

    for (float log_gamma = m_gamma_min;
	 log_gamma <= m_gamma_max;
	 log_gamma += m_gamma_step)
    {
	// LOO cross validation
	double loo_error;

	if (m_verbose) {
	    trainer.set_search_lambdas(logspace(-9, 4, 100));
	    trainer.be_verbose();
	}
	double gamma = exp10_ (log_gamma);
	trainer.set_kernel (Kernel_type (gamma));
#if DLIB_REVISION == 4093
        /* dlib 17.34 */
	dlib::decision_function<Kernel_type> krr_df = 
	    trainer.train (m_samples, m_labels, loo_error);
#elif DLIB_MAJOR_VERSION == 17 && DLIB_MINOR_VERSION == 44
        /* dlib 17.44 */
        /* GCS FIX: How to get loo_error from loo_values? */
        std::vector<double> loo_values;
	double lambda_used;
	dlib::decision_function<Kernel_type> krr_df = 
            trainer.train (m_samples, m_labels, loo_values, lambda_used);
#else
        error, unknown DLIB version!;
#endif

	if (loo_error < best_loo) {
	    best_loo = loo_error;
	    best_gamma = gamma;
	    m_krr_df = krr_df;
	}
	printf ("10^%f %9.6f\n", log_gamma, loo_error);
    }

    printf ("Best result: gamma=10^%f (%g), loo_error=%9.6f\n",
	log10(best_gamma), best_gamma, best_loo);
}
