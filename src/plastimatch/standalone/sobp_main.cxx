/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bragg_curve.h"
#include "ion_sobp_optimize.h"

int main (int argc, char* argv[])
{
    double dmin, dmax;
    double Emin, Emax;
    char choice;

    Ion_sobp_optimize test;

    if (argc < 4) {
        printf (
            "Usage:\n"
            "  sobp d dmin dmax    // Optimize in mm from dmin to dmax\n"
            "  sobp e emin emax    // Optimize in MeV from emin to emax\n");
        exit (0);
    }

    // construction of the sobp using the proximal and distal limits
    if (argv[1][0]=='d')
    {
        sscanf (argv[2], "%lf", &dmin);
        sscanf (argv[3], "%lf", &dmax);
        test.SetMinMaxDepths(dmin, dmax);
    }
    // construction of the sobp using the lower and higher energy
    else if (argv[1][0]=='e')
    {
        sscanf (argv[2], "%lf", &Emin);
        sscanf (argv[3], "%lf", &Emax);
        test.SetMinMaxDepths(Emin, Emax);
    }

    test.printparameters();
    test.Optimizer();

    // give the choice for returning the optimized weights, 
    // the sobp depth dose, or both of them
    printf("\n Do you want to see the peak weights (1), sobp output (2) or both (3)? ");
    scanf("%c", &choice);
    if (choice =='1')
    {
        test.SobpOptimizedWeights();
    }
    else if (choice =='2')
    {
        test.SobpDepthDose();
    }
    else if (choice =='3')
    {
        test.SobpOptimizedWeights();
        test.SobpDepthDose();
    }

    return 0;
}
