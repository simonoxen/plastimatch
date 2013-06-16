/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
   Sobp (Spread Out Bragg Peak) is a class that creates a sobp from minimal and
   maximal depth to be targeted, or minimal and maximal energies of the
   pristine Bragg Peaks used to create the sobp. This class return the 
   weights of each pristine peak or the created sobp. It contains also the
   optimization process to get the weight of the pristine peaks.
   ----------------------------------------------------------------------- */

#ifndef _sobp_h_
#define _sobp_h_

#include "plmdose_config.h"

#include <vector>
#include "bragg_curve.h"

/* the sobp function are wrapped in a class */
class PLMDOSE_API Sobp
{
public:
    Sobp();
    ~Sobp();
    void SetMinMaxEnergies(int E_min, int E_max);			// set the minimal and maximal energy to buld the sobp peak
    void SetMinMaxEnergies(int E_min, int E_max, int step); // set the minimal and maximal energy to buld the sobp peak and energy step
    void SetMinMaxDepths(int z_min, int z_max);				// set the minimal and maximal depth covered by the sobp
    void SetMinMaxDepths(int z_min, int z_max, int step);	// set the minimal and maximal depth covered by the sobp
    void SetEnergieStep(int step);							// set energy step
    void Optimizer();						// Weight optimizer
    void SobpOptimizedWeights();			// Return a matrix containing the weights of the pristine peaks
    void SobpDepthDose();		// Return a matrix containing the sobp depth-dose curve (depth, dose)
    void printparameters();

private:
    int m_E_min, m_E_max;				// Energies min and max of the Bragg Peak used to create the sobp, in MeV
    int m_EnergyStep;						// Step between the energies used to create the sobp, in MeV
    int m_EnergyNumber;				// number of energies used to create the sobp
    int m_z_min, m_z_max;			// Depth min and max to be reached by the sobp, in mm
    int m_z_end;						// Final depth of the matrix, in mm
    std::vector<double> m_weights;				// Matrix containing the weights and the associated BP energy
    std::vector<std::vector<double> > m_sobpDoseDepth;		    // Matrix containing the sobp depth dose curve
};

double cost_function_calculation(std::vector<std::vector<double> > depth_dose, std::vector<double> weights, int peak_number, int z_max, std::vector<int> depth_in, std::vector<int> depth_out);

#endif
