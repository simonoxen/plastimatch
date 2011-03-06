/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <iostream>
#include <stdio.h>
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

namespace itk {

Autolabel_ransac_est::Autolabel_ransac_est()
{
    this->deltaSquared = NumericTraits<double>::min();
    this->minForEstimate = 2;
    this->min_slope = -0.079;
    this->max_slope = -0.030;
}

Autolabel_ransac_est::~Autolabel_ransac_est()
{
}

void Autolabel_ransac_est::SetDelta( double delta )
{
    this->deltaSquared = delta*delta;
}

double Autolabel_ransac_est::GetDelta()
{
    return sqrt( this->deltaSquared );
}

void
Autolabel_ransac_est::set_slope_constraints (
    double min_slope, 
    double max_slope
)
{
    this->min_slope = min_slope;
    this->max_slope = max_slope;
}

/* parameters[0] is slope, parameters[1] is offset */
void
Autolabel_ransac_est::Estimate (
    std::vector<Autolabel_point *> &data, 
    std::vector<double> &parameters
)
{
    const double EPS = 2*NumericTraits<double>::epsilon(); 

    /* Make sure we have two points */
    parameters.clear();
    if (data.size() < this->minForEstimate) {
	return;
    }

    /* Check x displacement is sufficient */
    Autolabel_point& p1 = *(data[0]);
    Autolabel_point& p2 = *(data[1]);
    double xdiff = p2[0] - p1[0];
    if (fabs(xdiff) < 10 * EPS) {
	return;
    }

    /* Compute slope */
    double ydiff = p2[1] - p1[1];
    double slope = ydiff / xdiff;
    double intercept = p1[1] - slope * p1[0];
    if (slope < this->min_slope || slope > this->max_slope) {
#if defined (commentout)
	printf ("slope failed (%f,%f) (%f,%f) s=%f, i=%f\n",
	    p1[0],p1[1],p2[0],p2[1],slope,intercept);
#endif
	return;
    }

    /* Compute intercept */

    parameters.push_back (slope);
    parameters.push_back (intercept);
}

/* WTF? */
void 
Autolabel_ransac_est::Estimate (
    std::vector< Autolabel_point > &data, 
    std::vector<double> &parameters
)
{
    std::vector< Autolabel_point *> usedData;
    int dataSize = data.size();
    for( int i=0; i<dataSize; i++ )
	usedData.push_back( &(data[i]) );
    Estimate( usedData, parameters );
}


void 
Autolabel_ransac_est::LeastSquaresEstimate (
    std::vector< Autolabel_point *> &data, 
    std::vector<double> &parameters
)
{
    /* Do something here */
}

/* WTF? */
void 
Autolabel_ransac_est::LeastSquaresEstimate (
    std::vector< Autolabel_point > &data, 
    std::vector<double> &parameters
)
{
    std::vector< Autolabel_point *> usedData;
    int dataSize = data.size();
    for( int i=0; i<dataSize; i++ )
	usedData.push_back( &(data[i]) );
    LeastSquaresEstimate( usedData, parameters );
}

bool
Autolabel_ransac_est::Agree (
    std::vector<double> &parameters, 
    Autolabel_point &data
)
{
    double slope = parameters[0];
    double intercept = parameters[1];

    double yest = intercept + slope * data[0];
    double ydiff = yest - data[1];
    return (ydiff*ydiff < this->deltaSquared);
}

} // end namespace itk
