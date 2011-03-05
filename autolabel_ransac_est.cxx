/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <iostream>
#include "autolabel_ransac_est.h"
#include "RANSAC.h"
#include "PlaneParametersEstimator.h"
#include "RandomNumberGenerator.h"

typedef itk::RANSAC< Autolabel_point, double> Ransac_type;

void
autolabel_ransac_est (Autolabel_point_vector& apv)
{
    Autolabel_point_vector tmp;
    std::vector<double> autolabel_parameters;

    itk::Autolabel_ransac_est::Pointer estimator 
	= itk::Autolabel_ransac_est::New();
    estimator->SetDelta (0.5);

    /* Run RANSAC */
    double desiredProbabilityForNoOutliers = 0.999;
    double percentageOfDataUsed;
    Ransac_type::Pointer ransac_estimator = Ransac_type::New();
    ransac_estimator->SetData (apv);
    ransac_estimator->SetParametersEstimator (estimator.GetPointer());
    percentageOfDataUsed = ransac_estimator->Compute (
	autolabel_parameters, desiredProbabilityForNoOutliers);
    if (autolabel_parameters.empty()) {
	std::cout<<"RANSAC estimate failed, degenerate configuration?\n";
	exit (-1);
    } else {
	printf ("RANSAC parameters: [s,i] = [%f,%f]\n", 
	    autolabel_parameters[0], autolabel_parameters[1]);
	printf ("Used %f percent of data.\n", percentageOfDataUsed);
    }

    /* Fill in 3rd component of apv with RANSAC estimates */
    Autolabel_point_vector::iterator it;
    double slope = autolabel_parameters[0];
    double intercept = autolabel_parameters[1];
    for (it = apv.begin(); it != apv.end(); it++) {
	double x = (*it)[0];
	//double y = (*it)[1];
	double ry = intercept + slope * x;
	(*it)[2] = ry;
    }
}
