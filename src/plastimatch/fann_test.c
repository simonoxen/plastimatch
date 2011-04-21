/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include "fann.h"

#define USE_SCALE 1

int
main (int argc, char* argv[])
{
    const unsigned int num_layers = 3;
    //const unsigned int num_neurons_hidden = 96;
    const unsigned int num_neurons_hidden = 3;
    const float desired_error = (const float) 0.001;
    struct fann *ann;
    struct fann_train_data *train_data, *test_data;
    unsigned int i = 0;

    if (argc != 2) {
	printf ("Usage: fann_test infile\n");
	exit (-1);
    }

    printf("Creating network.\n");

    train_data = fann_read_train_from_file (argv[1]);

    if (!train_data) {
	exit (-1);
    }

    ann = fann_create_standard (num_layers, train_data->num_input, 
	num_neurons_hidden, train_data->num_output);
    //fann_randomize_weights (ann, -1.0, +1.0);
    //fann_init_weights (ann, train_data);
    printf("Training network.\n");

    fann_set_training_algorithm (ann, FANN_TRAIN_BATCH);
    //fann_set_training_algorithm (ann, FANN_TRAIN_RPROP);
    //fann_set_training_algorithm (ann, FANN_TRAIN_QUICKPROP);
    //fann_set_learning_momentum (ann, 0.4);
    fann_set_activation_function_hidden (ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output (ann, FANN_LINEAR);

#if (USE_SCALE)
    fann_set_scaling_params (ann, train_data, -1.0, +1.0, -1.0, +1.0);
    fann_scale_train (ann, train_data);
#endif

    fann_train_on_data (ann, train_data, 10000, 10, desired_error);

    printf("Testing network.\n");

    /* Test on training data */
    test_data = fann_read_train_from_file (argv[1]);

    fann_reset_MSE(ann);

    for (i = 0; i < fann_length_train_data(test_data); i++)
    {
	fann_type *calc_out;

	fann_reset_MSE (ann);
#if (USE_SCALE)
	fann_scale_input (ann, test_data->input[i]);
#endif
	calc_out = fann_run (ann, test_data->input[i]);
#if (USE_SCALE)
	fann_descale_output (ann, calc_out);
#endif
	printf ("Result %f original %f error %f\n",
	    calc_out[0], test_data->output[i][0],
	    (float) fann_abs(calc_out[0] - test_data->output[i][0]));
    }
    //    printf ("MSE error on test test_data: %f\n", fann_get_MSE(ann));

    //printf ("Saving network.\n");
    fann_save (ann, "fann_test.net");

    //fann_print_connections(ann);
    //fann_print_parameters(ann);


    printf("Cleaning up.\n");
    fann_destroy_train (train_data);
    fann_destroy_train (test_data);
    fann_destroy (ann);

    return 0;
}
