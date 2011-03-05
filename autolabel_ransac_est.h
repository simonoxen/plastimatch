/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _autolabel_ransac_est_h_
#define _autolabel_ransac_est_h_

#include "plm_config.h"
#include "ParametersEstimator.h"
#include <itkObjectFactory.h>
#include "itk_point.h"

typedef DoublePoint3DType Autolabel_point;
typedef std::vector< Autolabel_point > Autolabel_point_vector;

namespace itk {

class Autolabel_ransac_est : 
	public ParametersEstimator< Autolabel_point, double > 
{
public:
    typedef Autolabel_ransac_est                              Self;
    typedef ParametersEstimator< Autolabel_point, double >  Superclass;
    typedef SmartPointer<Self>                                Pointer;
    typedef SmartPointer<const Self>                          ConstPointer;
 
    itkTypeMacro( Autolabel_ransac_est, ParametersEstimator );
    itkNewMacro( Self )

    virtual void Estimate( std::vector< Autolabel_point *> &data, 
	std::vector<double> &parameters );
    virtual void Estimate( std::vector< Autolabel_point > &data, 
	std::vector<double> &parameters );

    virtual void LeastSquaresEstimate( std::vector< Autolabel_point *> &data, 
	std::vector<double> &parameters );
    virtual void LeastSquaresEstimate( std::vector< Autolabel_point > &data, 
	std::vector<double> &parameters );

    virtual bool Agree( std::vector<double> &parameters, 
	Autolabel_point &data );
  
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

plastimatch1_EXPORT
void
autolabel_ransac_est (Autolabel_point_vector& apv);

#endif
