/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _autolabel_ransac_est_h_
#define _autolabel_ransac_est_h_

#include "plm_config.h"
#include "ParametersEstimator.h"
#include <itkObjectFactory.h>
#include "itk_point.h"

namespace itk {

class Autolabel_ransac_est : 
	public ParametersEstimator< DoublePoint2DType, double > 
{
public:
    typedef Autolabel_ransac_est                              Self;
    typedef ParametersEstimator< DoublePoint2DType, double >  Superclass;
    typedef SmartPointer<Self>                                Pointer;
    typedef SmartPointer<const Self>                          ConstPointer;
 
    itkTypeMacro( Autolabel_ransac_est, ParametersEstimator );
    itkNewMacro( Self )

    virtual void Estimate( std::vector< DoublePoint2DType *> &data, 
	std::vector<double> &parameters );
    virtual void Estimate( std::vector< DoublePoint2DType > &data, 
	std::vector<double> &parameters );

    virtual void LeastSquaresEstimate( std::vector< DoublePoint2DType *> &data, 
	std::vector<double> &parameters );
    virtual void LeastSquaresEstimate( std::vector< DoublePoint2DType > &data, 
	std::vector<double> &parameters );

    virtual bool Agree( std::vector<double> &parameters, 
	DoublePoint2DType &data );
  
    void SetDelta( double delta );
    double GetDelta();
    void set_slope_constraints (double min_slope, double max_slope);

protected:
    Autolabel_ransac_est();
    ~Autolabel_ransac_est();

private:
    Autolabel_ransac_est(const Self& ); //purposely not implemented
    void operator=(const Self& ); //purposely not implemented
    double deltaSquared; 
    double min_slope;
    double max_slope;
};

} // end namespace itk

#include "autolabel_ransac_est.txx"

#endif
