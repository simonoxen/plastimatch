/*
    This is a command line program that can try different regression 
    algorithms on a libsvm-formatted data set.
*/
#include <iostream>
#include <map>
#include <vector>

#include "dlib/cmd_line_parser.h"

typedef dlib::cmd_line_parser<char>::check_1a_c Clp;

/* ---------------------------------------------------------------------
   global functions
   --------------------------------------------------------------------- */
static void
print_usage (Clp& parser)
{
    std::cout << "Usage: dlib_test [options] --in input_file [file ...]\n";
    parser.print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_args (Clp& parser, int argc, char* argv[])
{
    try {
	// Algorithm-independent options
        parser.add_option ("a","",1);
        parser.add_option ("algorithm",
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
	    print_usage (parser);
	    exit (0);
        }

	// Check that an input file was given
        if (!parser.option("in")) {
	    std::cout << 
		"Error: "
		"You must specify an input file with the --in option.\n";
	    print_usage (parser);
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
    for (int i =0; i < parser.number_of_arguments(); ++i) {
	std::cout << "unlabeled option " << i << " is " 
		  << parser[i]
		  << std::endl;
    }
}

int 
main (int argc, char* argv[])
{
    Clp parser;

    parse_args(parser, argc, argv);

    const Clp::option_type& option_alg = parser.option("a");
    const Clp::option_type& option_in = parser.option("in");

    // Normalize inputs to N(0,1)
    if (parser.option ("normalize")) {
    }

    if (!option_alg) {
	// Do KRR if user didn't specify an algorithm
	std::cout << "No algorithm specified, default to KRR\n";
    }
    else if (option_alg.argument() == "krls") {
	printf ("Hello world\n");
    }
    else if (option_alg.argument() == "krr") {
    }
    else if (option_alg.argument() == "mlp") {
    }
    else if (option_alg.argument() == "svr") {
    }
    else {
	fprintf (stderr, 
	    "Error, algorithm \"%s\" is unknown.\n"
	    "Please use -h to see the command line options\n",
	    option_alg.argument().c_str());
	exit (-1);
    }
}
