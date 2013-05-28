#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "bragg_curve.h"
#include "sobp.h"

int main (int argc, char* argv[])
{
	int dmin, dmax;
	int Emin, Emax;
	char choice;

	Sobp test;

	if (argv[1][0]=='d') // construction of the sobp using the proximal and distal limits
	{
		sscanf (argv[2], "%d", &dmin);
		sscanf (argv[3], "%d", &dmax);
		test.SetMinMaxDepths(dmin, dmax);
	}
	else if (argv[1][0]=='e') // construction of the sobp using the lower and higher energy
	{
		sscanf (argv[2], "%d", &Emin);
		sscanf (argv[3], "%d", &Emax);
		test.SetMinMaxDepths(Emin, Emax);
	}

	test.printparameters();
	test.Optimizer();

	printf("\n Do you want to see the peak weights (1), sobp output (2) or both (3)?"); // give the choice for returning the optimized weights, the sobp depth dose or both of them on the command line
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
