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


void Rt_sobp::Optimizer ()
{
    d_ptr->num_peaks = (int)(((d_ptr->E_max - d_ptr->E_min) / d_ptr->eres) + 1);

	double depth_maximum = 0;
	int idx_maximum = 0;

    std::vector<int> energies (d_ptr->num_peaks,0);
    std::vector<double> weight (d_ptr->num_peaks, 0);
  
    std::vector<double> init_vector (d_ptr->num_samples, 0);
    std::vector< std::vector<double> > depth_dose (d_ptr->num_peaks, init_vector);

    printf("\n %d Mono-energetic BP used:\n", d_ptr->num_peaks);

    /* updating the energies in the table) */
    for (int i = 0; i < d_ptr->num_peaks; i++)
    {
        energies[i]= d_ptr->E_min + i * d_ptr->eres;
        printf("%d ", energies[i]);
    }
    printf("\n");

    for (int i = 0; i < d_ptr->num_peaks; i++)
    {
        for (int j = 0; j < d_ptr->num_samples; j++)
        {
            depth_dose[i][j] = bragg_curve_norm((double)energies[i],1,(double)d_ptr->d_lut[j]);
        }
    }
	
    for (int i = d_ptr->num_peaks -1 ; i >= 0; i--)
    {
        if (i == d_ptr->num_peaks - 1)
        {
            weight[i] = 1.0;
        }
        else
        {
			/* Find depth max in mm*/
			depth_maximum = (double) get_proton_depth_max(energies[i]) / (100 * d_ptr->dres);
			idx_maximum = (int) depth_maximum;

			if (depth_maximum - (double) ((int) depth_maximum) > 0.5 &&  idx_maximum < d_ptr->num_samples)
			{
				idx_maximum++;
			}
            weight[i] = 1.0 - d_ptr->e_lut[idx_maximum];
            if (weight[i] < 0)
            {
                weight[i] = 0;
            }
        }
        for (int j = 0; j < d_ptr->num_samples; j++)
        {
            d_ptr->e_lut[j] += weight[i] * depth_dose[i][j];
        }
    }

	double mean_sobp = 0;
	double mean_count = 0;
	for (int i = 0; i < d_ptr->num_samples; i++)
	{
		if (d_ptr->d_lut[i] >= d_ptr->dmin && d_ptr->d_lut[i] <= d_ptr->dmax)
		{
			mean_sobp += d_ptr->e_lut[i];
			mean_count++;
		}
	}
	if (mean_count == 0)
	{
		printf("***WARNING*** The dose is null in the target interval\n");
		return;
	}

	/* SOBP norm and reset the depth dose*/
	for (int j = 0; j< d_ptr->num_samples; j++)
	{
		d_ptr->e_lut[j] =0;
	}

	for(int i = 0; i < d_ptr->num_peaks; i++)
	{
		weight[i] = weight[i] / mean_sobp * mean_count;
		for (int j = 0; j < d_ptr->num_samples; j++)
        {
            d_ptr->e_lut[j] += weight[i] * depth_dose[i][j];
        }
	}

    while (!d_ptr->depth_dose.empty())
    {
        d_ptr->depth_dose.pop_back();
    }

    d_ptr->num_peaks = d_ptr->num_peaks;
    for(int i = 0; i < d_ptr->num_peaks; i++)
    {
        this->add_peak ((double)energies[i],1, d_ptr->dres, 
            (double)d_ptr->dend, weight[i]);
        d_ptr->sobp_weight.push_back(weight[i]);
    }

    /* look for the max */
    double dose_max = 0;
    for(int i = d_ptr->num_samples-1; i >=0; i--)
    {
        if (d_ptr->e_lut[i] > dose_max)
        {
            dose_max = d_ptr->e_lut[i];
        }
    }
    this->SetDoseMax(dose_max);
}
