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
#ifndef _ion_sobp_optimize_h_
#define _ion_sobp_optimize_h_

#include "plmdose_config.h"
#include <vector>
#include "bragg_curve.h"

class PLMDOSE_API Ion_sobp_optimize
{
public:
    Ion_sobp_optimize();
    ~Ion_sobp_optimize();

    // set the minimal and maximal energy to buld the sobp peak
    void SetMinMaxEnergies(double E_min, double E_max);
    // set the minimal and maximal energy to buld the sobp peak and energy step
    void SetMinMaxEnergies(double E_min, double E_max, double step);
    // set the minimal and maximal depth covered by the sobp
    void SetMinMaxDepths(double z_min, double z_max);
    // set the minimal and maximal depth covered by the sobp
    void SetMinMaxDepths(double z_min, double z_max, double step);
    // set energy step
    void SetEnergieStep(double step);
    // Weight optimizer
    void Optimizer();
    // Return a matrix containing the weights of the pristine peaks
    void SobpOptimizedWeights();
    // Return a matrix containing the sobp depth-dose curve (depth, dose)
    void SobpDepthDose();
    void printparameters();

public:
    // Energies min and max of the Bragg Peak used to create the sobp, in MeV
    double m_E_min, m_E_max;
    // Step between the energies used to create the sobp, in MeV
    double m_EnergyStep;
    // number of discrete energies used to create the sobp
    int m_EnergyNumber;
    // Depth min and max to be reached by the sobp, in mm
    double m_z_min, m_z_max;
    // Final depth of the matrix, in mm.  This is an integer, indicating 
    // the size of the array used during optimization.  Optimization 
    // only considers the curve shape at integer depth values.
    int m_z_end;
    // Matrix containing the weights and the associated BP energy
    std::vector<double> m_weights;
    // Matrix containing the sobp depth dose curve
    std::vector<std::vector<double> > m_sobpDoseDepth;
};

#endif
