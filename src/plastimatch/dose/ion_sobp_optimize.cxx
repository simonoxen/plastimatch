/* -----------------------------------------------------------------------
   Sobp (Spread Out Bragg Peak) is a class that creates a sobp from minimal and
   maximal depth to be targeted, or minimal and maximal energies of the
   pristine Bragg Peaks used to create the sobp. This class return the 
   weights of each pristine peak or the created sobp. It contains also the
   optimization process to get the weight of the pristine peaks.
   ----------------------------------------------------------------------- */
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ion_sobp_optimize.h"

double 
cost_function_calculation (
    std::vector<std::vector<double> > depth_dose, 
    std::vector<double> weights, 
    int peak_number, 
    int z_end, 
    std::vector<int> depth_in, 
    std::vector<int> depth_out);

// initialisation of the parameter using a default configuration, 
// a sobp constituted of 11 BP.
Ion_sobp_optimize::Ion_sobp_optimize()
{
    this->SetMinMaxEnergies (80.0, 90.0, 1.0);
}

Ion_sobp_optimize::~Ion_sobp_optimize()
{
}

// return on the command line the parameters of the sobp to be build
void Ion_sobp_optimize::printparameters()
{
    printf("E_min : %lf\n", m_E_min);
    printf("E_max : %lf\n", m_E_max);
    printf("E_EnergyStep : %lf\n", m_EnergyStep);
    printf("z_min : %lf\n", m_z_min);
    printf("z_max : %lf\n", m_z_max);
    printf("z_end : %d\n", m_z_end);
}

// set the sobp parameters by introducing the min and max energies
void Ion_sobp_optimize::SetMinMaxEnergies (double E_min, double E_max)
{
    this->SetMinMaxEnergies (E_min, E_max, this->m_EnergyStep);
}

// set the sobp parameters by introducing the min and max energies, 
// and the step between them
void Ion_sobp_optimize::SetMinMaxEnergies (
    double E_min, double E_max, double step)
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

// set the sobp parameters by introducing the proximal and distal distances
void Ion_sobp_optimize::SetMinMaxDepths (double z_min, double z_max)
{
    this->SetMinMaxDepths (z_min, z_max, this->m_EnergyStep);
}

// set the sobp parameters by introducing the proximal and distal 
// distances and the step between the energies to be used
void 
Ion_sobp_optimize::SetMinMaxDepths (
    double z_min, double z_max, double step)
{
    if (step < 0)
    {
        step = - step;
    }
    if (z_max <= 0 || z_min <= 0 )
    {
        printf("Error: The depth min and max of the Sobp must be positive!\n");
        printf("zmin = %lf, zmax = %lf\n", z_min, z_max);
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

        m_E_min = pow((m_z_min/0.022),(1/1.77));
        m_E_max = pow((m_z_max/0.022),(1/1.77))+1;
        m_z_end = m_z_max + 50;
        m_EnergyStep = step;
        m_EnergyNumber = (int)(((m_E_max-m_E_min-1)/m_EnergyStep)+2);
    }
}

// set the sobp parameters by introducing only step between energies
void Ion_sobp_optimize::SetEnergieStep (double step)
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

// the optimizer to get the optimized weights of the beams, 
// optimized by a cost function (see below)
void Ion_sobp_optimize::Optimizer()
{
    // calculation of the energies between E_max and E_min separated by the step
    int n = (int)(((m_E_max-m_E_min-1)/m_EnergyStep)+2);
    m_EnergyNumber = n;

    std::vector<int> energies (n,0);
    std::vector<int> z (m_z_end, 0);

    std::vector<double> init_vector (m_z_end,0);
    std::vector<std::vector<double> > depth_dose (n,init_vector);
    // "/n speeds up the calculation - 15 for the normalisation of 
    // the function to 15 and 50 because the BP are normalized closed 
    // to 50 when created by plastimatch)
    std::vector<double> weight (n,(double)(floor((15.0/(50*n))*1000)/1000));

    std::vector<double> weight_minimum (n,0);
    std::vector<double> weight_compare (n,0);
    bool compare = 0;
    std::vector<int> depth_in (m_z_end, 0);
    std::vector<int> depth_out (m_z_end, 0);

    printf("\n %d Mono-energetic BP used: ", n);

    energies[0]= m_E_min;

    // creation of the matrix gathering all the depth dose of the 
    // BP constituting the sobp
    depth_dose[0][0] = bragg_curve(energies[0],1,0); 

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

    // creation of the two intervals that represents the inner 
    // part of the sobp and the outer part
    for (int i = 0; i < m_z_end ; i++) 
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

    // we calculate the cost function for the first time
    SUPER_TOT = cost_function_calculation (depth_dose, weight, n, 
        m_z_end, depth_in, depth_out); 

    // at first try, the weights are the best we found by now...
    for (int k = 0; k < n; k++) 
    {
        weight_minimum[k] = weight[k];
    }

    // compare equals 0 only when the weights between the 2 last 
    // optimization using the cost function are equal - we are trapped 
    // in a minimum
    while (compare!=1) 
    {
        f_tot = SUPER_TOT;
        // we calculate if the cost function is reduce by changing
        // a little bit one weight after another
        for (int i = 0; i < n; i++) 
        {
            weight_min = 0;
  
            // fcost value??
            f_tot = cost_function_calculation (depth_dose,weight, n, 
                m_z_end, depth_in, depth_out); 

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
                // we try with a lower weight (that must be > 0)
                weight[i] = weight[i]-0.001; 
                f_tot = cost_function_calculation (depth_dose, weight, n, 
                    m_z_end, depth_in, depth_out);

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
            f_tot = cost_function_calculation (depth_dose, weight, n, m_z_end, 
                depth_in, depth_out);

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

            // in this step, we continue with the best weight 
            // set between w, w-d, and w+d
            weight[i] = weight[i]+weight_min*0.001; 
        } // and we do that for all the weights

        // then we check if the weights are different with respect to 
        // the last run, if yes, we continue, if no we are in a minimum, 
        // so we break the optimization
        compare = 1; 
        for (int k = 0; k < n ; k++) 
            if (weight_minimum[k] == weight_compare[k] && compare ==1)
            {
                compare = compare;
            }
            else
            {
                compare = 0;
            }

        // we copy the optimized parameters in the weight matrix
        for (int k = 0; k < n ; k++) 
        {
            weight[k] = weight_minimum[k];
            weight_compare[k] = weight[k];
        }
    }

    // we send the weight matrix to the member of the sobp class
    for (int i = 0; i < n; i++) 
    {
        m_weights.push_back(weight[i]);
    }

    // we send the depth dose matrix to the member of the sobp class
    double sum =0; 
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

// we print the weights on the command line
void Ion_sobp_optimize::SobpOptimizedWeights() 
{
    for (int i = 0; i < m_EnergyNumber; i++)
    {
        printf(" %f", m_weights[i]);
    }
    printf ("\n\n");
}

void Ion_sobp_optimize::SobpDepthDose() // we print the depth dose on the command line
{
    for (int j = 0; j < m_z_end; j++)
    {
        printf(" %f %f \n", m_sobpDoseDepth[0][j], m_sobpDoseDepth[1][j]);
    }
}


// cost function to be optimized in order to find the best weights and 
// fit a perfect sobp
double 
cost_function_calculation (
    std::vector<std::vector<double> > depth_dose, 
    std::vector<double> weights, 
    int peak_number, 
    int z_end, 
    std::vector<int> depth_in, 
    std::vector<int> depth_out) 
{
    std::vector<double> diff (z_end, 0);
    std::vector<double> excess (z_end, 0);
    std::vector<double> f (z_end, 0);
    double f_tot = 0;
    double sobp_max_diff = 0;
    double sum = 0;

    for (int j = 0; j < z_end; j++) // we fit the curve on all the depth
    {
        sum = 0;
        for (int k = 0; k < peak_number; k++)
        {
            sum = sum + weights[k]*depth_dose[k][j];
        }
        // first parameters: the difference sqrt(standard deviation) 
        // between the curve and the perfect sobp, in the sobp area
        diff[j] = depth_in[j] * fabs(sum-15); 
        if (diff[j] > sobp_max_diff)
        {
            // second parameters: the max difference between the curve and 
            // the perfect sobp, in the sobp area
            sobp_max_diff = diff[j];					
        }

        // first parameters: the excess difference sqrt(standard deviation) 
        // between the curve and the perfect sobp, out of the sobp area 
        // (we want it far lower that the sobp flat region
        excess[j] = depth_out[j] * (sum-15);
        if (excess[j] < 0)
        {
            excess[j] = 0;
        }
        // this 3 parameters are assessed, and weighted by 3 coefficient 
        // (to be optimized to get a beautiful sobp) and the value 
        // of the global function is returned
#if defined (commentout)
        /* GCS FIX:  This equation is wrong.  The max diff should be 
           added after the loop completes */
        f[j]= 0.5 * max + 0.05 * diff[j]*diff[j] + 0.1 * excess[j] * excess[j]; 
#endif
        f[j]= 0.05 * diff[j]*diff[j] + 0.1 * excess[j] * excess[j]; 
        f_tot = f_tot+f[j];
    }

    /* GCS: Add in the maximum difference factor.  Not sure quite how 
       to weigh correctly... */
    f_tot += 0.005 * z_end * sobp_max_diff;

    return f_tot; //we return the fcost value
}
