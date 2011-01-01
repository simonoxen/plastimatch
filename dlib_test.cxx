// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the kernel ridge regression 
    object from the dlib C++ Library.

    This example will train on data from the sinc function.

*/

#include <iostream>
#include <map>
#include <vector>

#include "dlib/svm.h"
#include "dlib/data_io.h"

using namespace std;
using namespace dlib;

int main()
{
    typedef std::map<unsigned long, double> sample_type;
    std::vector<sample_type> samples;
    std::vector<double> labels;

    load_libsvm_formatted_data (
        "/home/gsharp/Dropbox/autolabel/t-spine/t-spine.libsvm.all.scale", 
	samples, 
	labels
    );

#if defined (commentout)
    std::vector<sample_type>::iterator it;
    for (it = samples.begin(); 
	 it != samples.end();
	 it++)
    {
	sample_type& s = *it;
	sample_type::iterator map_it;
	for (map_it = s.begin();
	     map_it != s.end();
	     map_it++)
	{
	    unsigned long t = map_it->first;
	    std::cout << t << " (" << map_it->second << ") ";
	}
	std::cout << std::endl;
    }

    std::vector<double>::iterator labels_it;
    for (labels_it = labels.begin(); 
	 labels_it != labels.end();
	 labels_it++)
    {
	double t = (*labels_it);
	std::cout << t << std::endl;
    }
#endif

    std::vector<matrix<sample_type::value_type::second_type,0,1> > 
	dense_matrix;
    dense_matrix = sparse_to_dense (samples);

#if defined (commentout)
    typedef radial_basis_kernel<sample_type> kernel_type;
    krr_trainer<kernel_type> trainer;
#endif

#if defined (commentout)
    // Now we are making a typedef for the kind of kernel we want to use.  I picked the
    // radial basis kernel because it only has one parameter and generally gives good
    // results without much fiddling.
    typedef radial_basis_kernel<sample_type> kernel_type;

    // Here we declare an instance of the krr_trainer object.  This is the
    // object that we will later use to do the training.
    krr_trainer<kernel_type> trainer;

    // Here we set the kernel we want to use for training.   The radial_basis_kernel 
    // has a parameter called gamma that we need to determine.  As a rule of thumb, a good 
    // gamma to try is 1.0/(mean squared distance between your sample points).  So 
    // below we are using a similar value computed from at most 2000 randomly selected
    // samples.
    const double gamma = 3.0/compute_mean_squared_distance(randomly_subsample(samples, 2000));
    cout << "using gamma of " << gamma << endl;
    trainer.set_kernel(kernel_type(gamma));

    // now train a function based on our sample points
    decision_function<kernel_type> test = trainer.train(samples, labels);

    // now we output the value of the sinc function for a few test points as well as the 
    // value predicted by our regression.
    m(0) = 2.5; cout << sinc(m(0)) << "   " << test(m) << endl;
    m(0) = 0.1; cout << sinc(m(0)) << "   " << test(m) << endl;
    m(0) = -4;  cout << sinc(m(0)) << "   " << test(m) << endl;
    m(0) = 5.0; cout << sinc(m(0)) << "   " << test(m) << endl;

    // The output is as follows:
    //using gamma of 0.075
    //    0.239389   0.239389
    //    0.998334   0.998362
    //    -0.189201   -0.189254
    //    -0.191785   -0.186618

    // The first column is the true value of the sinc function and the second
    // column is the output from the krr estimate.  


    // Note that the krr_trainer has the ability to tell us the leave-one-out cross-validation
    // accuracy.  The train() function has an optional 3rd argument and if we give it a double
    // it will give us back the LOO error.
    double loo_error;
    trainer.train(samples, labels, loo_error);
    cout << "mean squared LOO error: " << loo_error << endl;
    // Which outputs the following:
    // mean squared LOO error: 8.29563e-07




    // Another thing that is worth knowing is that just about everything in dlib is serializable.
    // So for example, you can save the test object to disk and recall it later like so:
    ofstream fout("saved_function.dat",ios::binary);
    serialize(test,fout);
    fout.close();

    // now lets open that file back up and load the function object it contains
    ifstream fin("saved_function.dat",ios::binary);
    deserialize(test, fin);
#endif
}
