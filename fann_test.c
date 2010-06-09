/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include "fann.h"

int
main (int argc, char* argv[])
{
    const unsigned int num_layers = 3;
    const unsigned int num_neurons_hidden = 96;
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

    printf("Training network.\n");

    fann_set_training_algorithm (ann, FANN_TRAIN_INCREMENTAL);
    fann_set_learning_momentum (ann, 0.4);
    fann_train_on_data (ann, train_data, 3000, 10, desired_error);

    printf("Testing network.\n");

    /* Test on training data */
    test_data = fann_read_train_from_file (argv[1]);

    fann_reset_MSE(ann);
    for(i = 0; i < fann_length_train_data(test_data); i++)
    {
	fann_test(ann, test_data->input[i], test_data->output[i]);
    }
    printf ("MSE error on test data: %f\n", fann_get_MSE(ann));

    //printf ("Saving network.\n");
    //fann_save (ann, "robot_float.net");

    printf("Cleaning up.\n");
    fann_destroy_train (train_data);
    fann_destroy_train (test_data);
    fann_destroy (ann);

    return 0;
}
