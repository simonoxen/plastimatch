/* -----------------------------------------------------------------------
   Sobp (Spread Out Bragg Peak) is a class that creates a sobp from minimal and
   maximal depth to be targeted, or minimal and maximal energies of the
   pristine Bragg Peaks used to create the sobp. This class return the 
   weights of each pristine peak or the created sobp. It contains also the
   optimization process to get the weight of the pristine peaks.
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sobp.h"

Sobp::Sobp() // initialisation of the parameter using a default configuration, a sobp constituted of 11 BP.
{
	m_E_min = 80;
	m_E_max = 90;
	m_EnergyStep = 1;
	m_EnergyNumber = (int)(((m_E_max-m_E_min-1)/m_EnergyStep)+2);
	m_z_min = int((0.022)*pow(m_E_min, 1.77));
	m_z_max = int((0.022)*pow(m_E_max, 1.77))+1;
	m_z_end = m_z_max+50;
}

Sobp::~Sobp()
{

}

void Sobp::printparameters()  // return on the command line the parameters of the sobp to be build
{
	printf("E_min : %d\n",m_E_min);
	printf("E_max : %d\n",m_E_max);
	printf("E_EnergyStep : %d\n",m_EnergyStep);
	printf("z_min : %d\n",m_z_min);
	printf("z_max : %d\n",m_z_max);
	printf("z_end : %d\n",m_z_end);
}

void Sobp::SetMinMaxEnergies(int E_min, int E_max) // set the sobp parameters by introducing the min and max energies
{
	if (E_max <= 0 || E_min <= 0)
	{
		printf("The energies min and max of the Sobp must be positive!\n");
	}
	
	if (E_max >= E_min)
	{
		m_E_min = E_min;
		m_E_max = E_max;
	}
	else
	{
		m_E_min = E_max;
		m_E_max = E_min;
	}

	m_z_min = int((0.022)*pow(m_E_min, 1.77));
	m_z_max = int((0.022)*pow(m_E_max, 1.77))+1;
	m_z_end = m_z_max + 50;
	m_EnergyNumber = (int)(((m_E_max-m_E_min-1)/m_EnergyStep)+2);
}

void Sobp::SetMinMaxEnergies(int E_min, int E_max, int step) // set the sobp parameters by introducing the min and max energies, and the step between them
{
	if (E_max <= 0 || E_min <= 0)
		{
			printf("The energies min and max of the Sobp must be positive!\n");
		}
	
	if (E_max >= E_min)
	{
		m_E_min = E_min;
		m_E_max = E_max;
	}
	else
	{
		m_E_min = E_max;
		m_E_max = E_min;
	}

	m_z_min = int((0.022)*pow(m_E_min, 1.77));
	m_z_max = int((0.022)*pow(m_E_max, 1.77))+1;
	m_z_end = m_z_max + 50;
	
	m_EnergyStep = step;
	m_EnergyNumber = (int)(((m_E_max-m_E_min-1)/m_EnergyStep)+2);
}

void Sobp::SetMinMaxDepths(int z_min, int z_max) // set the sobp parameters by introducing the proximal and distal distances
{
	if (z_max <= 0 || z_min <= 0)
		{
			printf("Error: The depth min and max of the Sobp must be positive!\n");
			printf("zmin = %d, zmax = %d\n", z_min, z_max);
		}
	else
	{	
		if (z_max >= z_min)
		{
			m_z_min = z_min;
			m_z_max = z_max;
		}
		else
		{
			m_z_min = z_max;
			m_z_max = z_min;
		}

		m_E_min = int(pow((m_z_min/0.022),(1/1.77)));
		m_E_max = int(pow((m_z_max/0.022),(1/1.77)))+1;
		m_z_end = m_z_max + 50;
		m_EnergyNumber = (int)(((m_E_max-m_E_min-1)/m_EnergyStep)+2);
	}
}

void Sobp::SetMinMaxDepths(int z_min, int z_max, int step) // set the sobp parameters by introducing the proximal and distal distances and the step between the energies to be used
{
	if (step < 0)
	{
		step = - step;
	}
	if (z_max <= 0 || z_min <= 0 )
	{
		printf("Error: The depth min and max of the Sobp must be positive!\n");
		printf("zmin = %d, zmax = %d\n", z_min, z_max);
	}
	else if (step == 0)
	{
		printf("The step must be positive!\n");
	}
	else
	{	
		if (z_max >= z_min)
		{
			m_z_min = z_min;
			m_z_max = z_max;
		}
		else
		{
			m_z_min = z_max;
			m_z_max = z_min;
		}

		m_E_min = int(pow((m_z_min/0.022),(1/1.77)));
		m_E_max = int(pow((m_z_max/0.022),(1/1.77)))+1;
		m_z_end = m_z_max + 50;
		m_EnergyStep = step;
		m_EnergyNumber = (int)(((m_E_max-m_E_min-1)/m_EnergyStep)+2);
	}
}

void Sobp::SetEnergieStep(int step) // set the sobp parameters by introducing only step between energies
{
	if (step == 0)
	{
		printf("The step must be positive!\n");
	}
	else
	{
		if (step < 0)
		{
			step = - step;
		}
		m_EnergyStep = step;
		m_EnergyNumber = (int)(((m_E_max-m_E_min-1)/m_EnergyStep)+2);
	}
}

void Sobp::Optimizer() // the optimizer to get the optimized weights of the beams, optimized by a cost function (see below)
{
	int n = (int)(((m_E_max-m_E_min-1)/m_EnergyStep)+2); // calculation of the energies between E_max and E_min separated by the step
	m_EnergyNumber = n;

	std::vector<int> energies (n,0);
	std::vector<int> z (m_z_end, 0);

	std::vector<double> init_vector (m_z_end,0);
	std::vector<std::vector<double> > depth_dose (n,init_vector);
	std::vector<double> weight (n,(double)(floor((15.0/(50*n))*1000)/1000)); // "/n speeds up the calculation - 15 for the normalisation of the function to 15 and 50 because the BP are normalized closed to 50 when created by plastimatch)

	std::vector<double> weight_minimum (n,0);
    std::vector<double> weight_compare (n,0);
    bool compare = 0;
    std::vector<int> depth_in (m_z_end, 0);
	std::vector<int> depth_out (m_z_end, 0);

	printf("\n %d Mono-energetic BP used: ", n);

	energies[0]= m_E_min;
	depth_dose[0][0] = bragg_curve(energies[0],1,0);  // creation of the matrix gathering all the depth dose of the BP constituting the sobp

	for (int j = 0; j < m_z_end; j++)
	{
		depth_dose[0][j] = bragg_curve(energies[0],1,j);
		z[j] = j;
	}

	printf("%d ", energies[0]);
    for (int i=1; i < n-1; i++)
    {
        energies[i]=energies[i-1]+m_EnergyStep;
        printf("%d ",energies[i]);
		for (int j = 0; j < m_z_end; j++)
		{
			depth_dose[i][j] = bragg_curve(energies[i],1,j);
		}
    }
	energies[n-1]= m_E_max;
	printf("%d ", energies[n-1]);

	for (int j = 0; j < m_z_end; j++)
	{
		depth_dose[n-1][j] = bragg_curve(energies[n-1],1,j);
	}

    for (int i = 0; i < m_z_end ; i++) // creation of the two intervals that represents the inner part of the sobp and the outer part
    {

        if (z[i]>=m_z_min && z[i]<=m_z_max)
        {
                depth_in[i] = 1;
                depth_out[i] = 0;
        }
        else
        {
            depth_in[i] = 0;
            depth_out[i] = 1;
        }
    }

    double f_tot=0;
    double SUPER_TOT = 0;
    int weight_min = 0;

    SUPER_TOT = cost_function_calculation(depth_dose,weight, n, m_z_end, depth_in, depth_out); // we calculate the cost function for the first time

    for (int k = 0; k < n; k++) // at first try, the weights are the best we found by now...
    {
        weight_minimum[k] = weight[k];
    }

    while (compare!=1) // compare equals 0 only when the weights between the 2 last optimization using the cost function are equal - we are trapped in a minimum
    {
        f_tot = SUPER_TOT;
        for (int i = 0; i < n; i++) // we calculate if the cost function is reduce by changing a little bit one weight after another
        {
            weight_min = 0;
  
            f_tot = cost_function_calculation(depth_dose,weight, n, m_z_end, depth_in, depth_out); // fcost value??

            if (f_tot < SUPER_TOT)
            {
                SUPER_TOT = f_tot; // if it is better we save the solution!
                for (int k = 0; k < n; k++)
                {
                    weight_minimum[k] = weight[k];
                }
            }

            if (weight[i] > 0.0001) 
            {
                weight[i] = weight[i]-0.001; // we try with a lower weight (that must be > 0)
                f_tot = cost_function_calculation(depth_dose,weight, n, m_z_end, depth_in, depth_out);

                if (f_tot < SUPER_TOT) // idem
                {
                    SUPER_TOT = f_tot;
                    weight_min = -1;
                    for (int k = 0; k < n; k++)
                    {
                        weight_minimum[k] = weight[k];
                    }
                }
                weight[i] = weight[i]+0.001;
            }

            weight[i] = weight[i]+0.001;  // we try with a higher weight
            f_tot = cost_function_calculation(depth_dose,weight, n, m_z_end, depth_in, depth_out);

            if (f_tot < SUPER_TOT) // idem
            {
                SUPER_TOT = f_tot;
                weight_min = 1;
                for (int k = 0; k < n; k++)
                    {
                        weight_minimum[k] = weight[k];
                    }
            }
            weight[i] = weight[i]-0.001;

            weight[i] = weight[i]+weight_min*0.001; // in this step, we continue with the best weight set between w, w-d, and w+d
        } // and we do that for all the weights

        compare = 1; 
        for (int k = 0; k < n ; k++) // then we check if the weights are different with respect to the last run, if yes, we continue, if no we are in a minimum, so we break the optimization
        if (weight_minimum[k] == weight_compare[k] && compare ==1)
        {
            compare = compare;
        }
        else
        {
            compare = 0;
        }

        for (int k = 0; k < n ; k++) // we copy the optimized parameters in the weight matrix
        {
            weight[k] = weight_minimum[k];
            weight_compare[k] = weight[k];
        }
	}

     for (int i = 0; i < n; i++) // we send the weight matrix to the member of the sobp class
            {
				m_weights.push_back(weight[i]);
            }

	 double sum =0; // we send the depth dose matrix to the member of the sobp class
	 m_sobpDoseDepth.push_back(init_vector);
	 m_sobpDoseDepth.push_back(init_vector);
	 for (int j = 0; j < m_z_end; j++)
	 {
		 sum = 0;
		 for (int i = 0; i < n ; i++)
		 {
			sum = sum + weight[i]*depth_dose[i][j];
		 }
		 m_sobpDoseDepth[0][j] = j;
		 m_sobpDoseDepth[1][j] = sum;
	 }
}

void Sobp::SobpOptimizedWeights() // we print the weights on the command line
{
	for (int i = 0; i < m_EnergyNumber; i++)
	{
		printf(" %f", m_weights[i]);
	}
	printf ("\n\n");
}

void Sobp::SobpDepthDose() // we print the depth dose on the command line
{
	for (int j = 0; j < m_z_end; j++)
	{
		printf(" %f %f \n", m_sobpDoseDepth[0][j], m_sobpDoseDepth[1][j]);
	}
}


double cost_function_calculation(std::vector<std::vector<double> > depth_dose, std::vector<double> weights, int peak_number, int z_end, std::vector<int> depth_in, std::vector<int> depth_out) // cost function to be optimized in order to find the best weights and fit a perfect sobp
{
	std::vector<double> diff (z_end, 0);
	std::vector<double> excess (z_end, 0);
	std::vector<double> f (z_end, 0);
	double f_tot = 0;
	double max = 0;
	double sum = 0;

	for (int j = 0; j < z_end; j++) // we fit the curve on all the depth
    {
        sum = 0;
        for (int k = 0; k < peak_number; k++)
        {
            sum = sum + weights[k]*depth_dose[k][j];
        }
        diff[j] = depth_in[j] * fabs(sum-15); // first parameters: the difference sqrt(standard deviation) between the curve and the perfect sobp, in the sobp area
        if (diff[j] > max)
        {
            max = diff[j];					// second parameters: the max difference between the curve and the perfect sobp, in the sobp area
        }

		excess[j] = depth_out[j] * (sum-15);// first parameters: the excess difference sqrt(standard deviation) between the curve and the perfect sobp, out of the sobp area (we want it far lower that the sobp flat region
        if (excess[j] < 0)
        {
             excess[j] = 0;
        }
        f[j]= 0.5 * max + 0.05 * diff[j]*diff[j] + 0.1 * excess[j] * excess[j]; // this 3 parameters are assessed, and weighted by 3 coefficient (to be optimized to get a beautiful sobp) and the value of the global function is returned
        f_tot = f_tot+f[j];
	}
	return f_tot; //we return the fcost value
}
