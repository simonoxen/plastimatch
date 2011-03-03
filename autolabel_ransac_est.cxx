/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <vnl/algo/vnl_svd.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include "autolabel_ransac_est.h"

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
    std::vector<DoublePoint2DType *> &data, 
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
    DoublePoint2DType& p1 = *(data[0]);
    DoublePoint2DType& p2 = *(data[1]);
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
    std::vector< DoublePoint2DType > &data, 
    std::vector<double> &parameters
)
{
    std::vector< DoublePoint2DType *> usedData;
    int dataSize = data.size();
    for( int i=0; i<dataSize; i++ )
	usedData.push_back( &(data[i]) );
    Estimate( usedData, parameters );
}


void 
Autolabel_ransac_est::LeastSquaresEstimate (
    std::vector< DoublePoint2DType *> &data, 
    std::vector<double> &parameters
)
{
    /* Do something here */
}

/* WTF? */
void 
Autolabel_ransac_est::LeastSquaresEstimate (
    std::vector< DoublePoint2DType > &data, 
    std::vector<double> &parameters
)
{
    std::vector< DoublePoint2DType *> usedData;
    int dataSize = data.size();
    for( int i=0; i<dataSize; i++ )
	usedData.push_back( &(data[i]) );
    LeastSquaresEstimate( usedData, parameters );
}

bool
Autolabel_ransac_est::Agree (
    std::vector<double> &parameters, 
    DoublePoint2DType &data
)
{
    double slope = parameters[0];
    double intercept = parameters[1];

    double yest = intercept + slope * data[0];
    double ydiff = yest - data[1];
    return (ydiff*ydiff < this->deltaSquared);
}

} // end namespace itk
