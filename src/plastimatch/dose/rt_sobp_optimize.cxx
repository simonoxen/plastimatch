/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include <math.h>

#include "bragg_curve.h"
#include "file_util.h"
#include "path_util.h"
#include "print_and_exit.h"
#include "rt_depth_dose.h"
#include "rt_sobp.h"
#include "rt_sobp_p.h"
#include "string_util.h"
#include "vnl/algo/vnl_amoeba.h"
#include "vnl/vnl_cost_function.h"

/* cost function used to optimize the sobp shape */
double cost_function_calculation(std::vector<std::vector<double> > depth_dose, std::vector<double> weights, int num_peaks, int num_samples, std::vector<int> depth_in, std::vector<int> depth_out);

/* vxl needs you to wrap the function within a class */
class cost_function : public vnl_cost_function
{
public:
    std::vector<std::vector<double> > depth_dose;
    std::vector<double> weights;
    std::vector<int> depth_in;
    int num_peaks;
    int num_samples;
    double z_end;
    std::vector<int> depth_out;

    virtual double f (vnl_vector<double> const& vnl_x) {
        /* vxl requires you using their own vnl_vector type, 
           therefore we copy into a standard C/C++ array. */
        for (int i=0; i < num_peaks; i++)
        {
            weights[i] =vnl_x[i];
        }
        return cost_function_calculation (depth_dose,weights, 
            num_peaks, num_samples, depth_in, depth_out);
    }
};

void Rt_sobp::Optimizer() // the optimizer to get the optimized weights of the beams, optimized by a cost function (see below)
{
    double E_max = 0;
    /* Create function object (for function to be minimized) */
    cost_function cf;

    cf.num_samples = d_ptr->num_samples;
    cf.num_peaks = d_ptr->num_peaks;
	
    for (int i = 0; i < d_ptr->num_peaks; i++)
    {
        cf.weights.push_back(0);
    }
	
    std::vector<int> energies (d_ptr->num_peaks,0);
    std::vector<double> init_vector (d_ptr->num_samples,0);


    cf.depth_dose.push_back(init_vector);

    printf("\n %d Mono-energetic BP used: ", cf.num_peaks);

    energies[0]= d_ptr->E_min;
    printf("%d ", energies[0]);

    cf.depth_dose[0][0] = bragg_curve((double)energies[0],1,0);  // creation of the matrix gathering all the depth dose of the BP constituting the sobp

    for (int j = 0; j < d_ptr->num_samples; j++)
    {
        cf.depth_dose[0][j] = bragg_curve((double)energies[0],1,(double)d_ptr->d_lut[j]);
        if (cf.depth_dose[0][j] > E_max)
        {
            E_max = cf.depth_dose[0][j];
        }
    }
    for (int j = 0; j < d_ptr->num_samples; j++) // we normalize the depth dose curve to 1
    {
        cf.depth_dose[0][j] = cf.depth_dose[0][j] / E_max;
    }


    for (int i=1; i < cf.num_peaks-1; i++)
    {
        energies[i]=energies[i-1]+d_ptr->eres;
        printf("%d ",energies[i]);
		
        cf.depth_dose.push_back(init_vector);
        E_max = 0;

        for (int j = 0; j < d_ptr->num_samples; j++)
        {
            cf.depth_dose[i][j] = bragg_curve(energies[i],1,d_ptr->d_lut[j]);
            if (cf.depth_dose[i][j] > E_max)
            {
                E_max = cf.depth_dose[i][j];
            }
        }
        for (int j = 0; j < d_ptr->num_samples; j++) // we normalize the depth dose curve to 1
        {
            cf.depth_dose[i][j] = cf.depth_dose[i][j] / E_max;
        }
    }

    energies[cf.num_peaks-1]= d_ptr->E_max;
    printf("%d \n", energies[cf.num_peaks-1]);

    cf.depth_dose.push_back(init_vector);
    for (int j = 0; j < d_ptr->num_samples; j++)
    {
        cf.depth_dose[cf.num_peaks-1][j] = bragg_curve(energies[cf.num_peaks-1],1,d_ptr->d_lut[j]);
    }


    for (int i = 0; i < d_ptr->num_samples ; i++) // creation of the two intervals that represents the inner part of the sobp and the outer part
    {
        cf.depth_in.push_back(0);
        cf.depth_out.push_back(0);

        if (d_ptr->d_lut[i]>=d_ptr->dmin && d_ptr->d_lut[i]<=d_ptr->dmax)
        {
            cf.depth_in[i] = 1;
            cf.depth_out[i] = 0;
        }
        else
        {
            cf.depth_in[i] = 0;
            cf.depth_out[i] = 1;
        }
    }	

    /* Create optimizer object */
    vnl_amoeba nm (cf);


    /* Set some optimizer parameters */
    nm.set_x_tolerance (0.0001);
    nm.set_f_tolerance (0.0000001);
    nm.set_max_iterations (1000000);

    /* Set the starting point */
    vnl_vector<double> x(cf.num_peaks, 1.0 / (double) cf.num_peaks);
    const vnl_vector<double> y(cf.num_peaks, 0.01 / (double) cf.num_peaks);

    /* Run the optimizer */
    nm.minimize (x,y);

    while (!d_ptr->depth_dose.empty())
    {
        d_ptr->depth_dose.pop_back();
    }

    for(int i = 0; i < d_ptr->num_peaks; i++)
    {
        this->add_peak ((double)energies[i],1, 
            d_ptr->dres, (double)d_ptr->dend, cf.weights[i]);
        d_ptr->sobp_weight.push_back(cf.weights[i]);
    }

    d_ptr->num_samples = d_ptr->depth_dose[0]->num_samples;

    this->generate();
}

void Rt_sobp::Optimizer2() // the optimizer to get the optimized weights of the beams, optimized by a cost function (see below)
{
    double dose_max = 0;
    /* Create function object (for function to be minimized) */

    int num_samples = d_ptr->num_samples;
    int num_peaks = d_ptr->num_peaks;
    std::vector<double> weight (num_peaks, 0);
    int depth_max = 0;
	
    std::vector<int> energies (num_peaks,0);
    std::vector<double> init_vector (num_samples,0);
    std::vector< std::vector<double> > depth_dose (num_peaks, init_vector);

    printf("\n %d Mono-energetic BP used:\n", num_peaks);

    for (int i = 0; i < num_peaks; i++)
    {
        energies[i]= d_ptr->E_min + i * d_ptr->eres;
        printf("%d ", energies[i]);
    }

    for (int i = 0; i < d_ptr->num_peaks; i++)
    {
        dose_max = 0;

        for (int j = 0; j < num_samples; j++)
        {
            depth_dose[i][j] = bragg_curve((double)energies[i],1,(double)d_ptr->d_lut[j]);
		
            if (depth_dose[i][j] > dose_max)
            {
                dose_max = depth_dose[i][j];
            }
        }

        for (int j = 0; j < num_samples; j++)
        {
            depth_dose[i][j] = depth_dose[i][j] / dose_max;
        }
    }

    for (int i = num_peaks -1 ; i >= 0; i--)
    {
        if (i == num_peaks - 1)
        {
            weight[i] = 1.0;
        }
        else
        {
            depth_max = max_depth_proton[ energies[i] ];
            weight[i] = 1.0 - d_ptr->e_lut[depth_max];
            if (weight[i] < 0)
            {
                weight[i] = 0;
            }
        }

        for (int j = 0; j < num_samples; j++)
        {
            d_ptr->e_lut[j] += weight[i] * depth_dose[i][j];
        }
    }

    for (int i = 0; i < 100; i++)
    {
        for (int i = 0; i < num_peaks; i++)
	{
            depth_max = max_depth_proton[ energies[i] ];
            weight[i] = weight[i] / d_ptr->e_lut[depth_max];
	}

	for (int j = 0 ; j < num_samples; j++)
	{
            d_ptr->e_lut[j] = 0;
            for (int i = 0; i < num_peaks; i++)
            {
                d_ptr->e_lut[j] += weight[i] * depth_dose[i][j];
            }
	}
    }

    while (!d_ptr->depth_dose.empty())
    {
        d_ptr->depth_dose.pop_back();
    }

    for(int i = 0; i < d_ptr->num_peaks; i++)
    {
        this->add_peak ((double)energies[i],1, d_ptr->dres, 
            (double)d_ptr->dend, weight[i]);
        d_ptr->sobp_weight.push_back(weight[i]);
    }

    d_ptr->num_samples = d_ptr->depth_dose[0]->num_samples;

    //this->generate();
}

double cost_function_calculation(std::vector<std::vector<double> > depth_dose, std::vector<double> weights, int num_peaks, int num_samples, std::vector<int> depth_in, std::vector<int> depth_out) // cost function to be optimized in order to find the best weights and fit a perfect sobp
{
    std::vector<double> diff (num_samples, 0);
    std::vector<double> excess (num_samples, 0);
    std::vector<double> f (num_samples, 0);
    double f_tot = 0;
    double sobp_max = 0;
    double sum = 0;

    for (int j = 0; j < num_samples; j++) // we fit the curve on all the depth
    {
        sum = 0;
        for (int k = 0; k < num_peaks; k++)
        {
            sum = sum + weights[k]*depth_dose[k][j];
        }
        diff[j] = depth_in[j] * fabs(sum-1); // first parameters: the difference sqrt(standard deviation) between the curve and the perfect sobp, in the sobp area
        if (diff[j] > sobp_max)
        {
            sobp_max = diff[j];					// second parameters: the max difference between the curve and the perfect sobp, in the sobp area
        }

        excess[j] = depth_out[j] * (sum-1);// first parameters: the excess difference sqrt(standard deviation) between the curve and the perfect sobp, out of the sobp area (we want it far lower that the sobp flat region
        if (excess[j] < 0)
        {
            excess[j] = 0;
        }
        f[j]= 0.05 * diff[j]*diff[j] + 0.1 * excess[j] * excess[j]; // this 3 parameters are assessed, and weighted by 3 coefficient (to be optimized to get a beautiful sobp) and the value of the global function is returned
        f_tot = f_tot+f[j];
    }

    f_tot += 0.005 * sobp_max * num_samples;

    for(int i=0; i < num_peaks; i++)
    {
        if (weights[i] < 0)
        {
            f_tot = 2* f_tot;
        }
    }
    /*printf("\n f_tot = %lg", f_tot);
      for (int i = 0; i < num_peaks; i++)
      {
      printf (" %lg ", weights[i]);
      }*/

    return f_tot; //we return the fcost value
}

