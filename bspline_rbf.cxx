/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "bspline.h"
#include "bspline_landmarks.h"
#include "bspline_rbf.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "volume.h"

#include <iostream>
#include "vnl/vnl_matrix_fixed.h"
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector.h"
#include "vnl/vnl_vector_fixed.h"
#include "vnl/algo/vnl_svd.h"
#include "vnl/vnl_sample.h"

typedef struct rbf_params Rbf_parms;
struct rbf_params { // used to pass information to bspline_rbf_score
    float radius;    // radius of the RBF
    Bspline_parms *bparms;
    Volume *vector_field;
};

/* Service routine for Nelder-Mead optimizer
Sort vertices of the simplex according to the value of the function */
void reorder_simplex( float *f, float *vertices,  int n )
{
    int i,j,k;
    float val;
    float *vert_tmp;

    vert_tmp = (float *)malloc(n*sizeof(float));

    //pick sorting; see e.g. Numerical Recipes 8.1
    for(j=1;j<n+1;j++)
    {
	val = f[j];
	for(k=0;k<n;k++) vert_tmp[k]=vertices[k+j*n];
	i = j-1;
	while( (i>-1) && (f[i]>val) ) { f[i+1]=f[i]; for(k=0;k<n;k++) vertices[k+(i+1)*n] = vertices[k+i*n];  i--; }
	f[i+1]=val;
	for(k=0;k<n;k++) vertices[k+(i+1)*n] = vert_tmp[k];
    }
 
    free(vert_tmp);
}
 

void print_coords(float *x, int n)
{
    int i;
    for(i=0;i<n;i++) printf("%.5f ",x[i]);
    printf("\n");
}

/* 
Nelder-Mead simplex method for multidimensional minimization, see
http://en.wikipedia.org/wiki/Nelder–Mead_method
Does not require derivatives

Input: 
n is the dimension
x are n*(n+1) trial simplex coordinates for n+1 vertices
func(float * = input vector, n = dimension, void * = parameters) will be minimized
params are parameters passed through to the function func

Output:
On exit, first n components of x is the minimum found, 
funcval[0] is the value of function at the minimum
*/
void minimize_nelder_mead (
    float *x, 
    float *funcval, 
    int n, 
    float(*func) (float *, int, void *), 
    void *params
)
{
    int iter=0;
    int i,j;
    float *x0, *xr, *xe, *xc;
    float val_xr, val_xe, val_xc;
    float alpha=1., gamma=2., rho=0.5, sigma=0.5;

    x0 = (float *)malloc( n*sizeof(float) );
    xr = (float *)malloc( n*sizeof(float) );
    xe = (float *)malloc( n*sizeof(float) );
    xc = (float *)malloc( n*sizeof(float) );

    //initialize
    for(i=0;i<n+1;i++)  funcval[i] = func( x+i*n, n, params );
    reorder_simplex( funcval, x, n );

    for(iter = 0; iter<100000; iter++)
    {
	// StepOne is here
	reorder_simplex( funcval, x, n );

	if ( (iter>0) && ( funcval[0]<1e-3) ) break;

	for(i=0;i<n;i++) x0[i]=0;

	for(i=0;i<n;i++)
	    for(j=0;j<n;j++) x0[i] += x[i+j*n];
	for(i=0;i<n;i++) x0[i]/=n; 

	//reflection
	for(i=0;i<n;i++) xr[i]=x0[i]+alpha*( x0[i] -x[i + n*n] );
	val_xr = func( xr, n, params );
	if  ( ( funcval[0] <= val_xr ) && (val_xr < funcval[n-1]) )
	{ for(i=0;i<n;i++) x[i+n*n] = xr[i]; funcval[n]=val_xr; continue; /*goto StepOne; */ }

	//expansion
	if ( val_xr < funcval[0])
	{
	    for(i=0;i<n;i++) xe[i]=x0[i]+gamma*( x0[i] -x[i+n*n] );
	    val_xe = func(xe, n, params);
	    if (val_xe < val_xr) { for(i=0;i<n;i++) x[i+n*n] = xe[i];  funcval[n]=val_xe; continue; /*  goto StepOne; */ } 
	    else { for(i=0;i<n;i++) x[i+n*n] = xr[i]; funcval[n]=val_xr; continue; /* goto StepOne; */}
	}
	/*else do nothing, goto contraction*/ 

	//contraction
	if ( val_xr < funcval[n-1]) { fprintf(stderr,"error in Nelder-Mead minimization!\n"); exit(1); }
	for(i=0;i<n;i++) xc[i]=x[i+n*n]+rho*( x0[i]-x[i+n*n] );
	val_xc = func(xc, n, params);
	if (val_xc < funcval[n]) { for(i=0;i<n;i++) x[i+n*n] = xc[i]; funcval[n]=val_xc; continue; /* goto StepOne; */ }

	//reduction
	for(i=1;i<n+1;i++)
	    for(j=0;j<n;j++) x[j+i*n] = x[j+0*n] + sigma * (x[j+i*n]-x[j+0*n]);
	for(i=0;i<n+1;i++)  funcval[i] = func( x+i*n, n, params );

	//goto StepOne;
    }

    //reorder so the first vertex is the best one
    reorder_simplex(funcval, x, n);

    free(x0); free(xr); free(xc); free(xe);
}


// radial basis function
// center and x,y,z are in voxels
// radius is in mm
float rbf_value (int *center, int x, int y, int z, 
    float radius, float *pix_spacing)
{
    float val, r, dx,dy,dz;

    dx = (center[0]-x)*pix_spacing[0];
    dy = (center[1]-y)*pix_spacing[1];
    dz = (center[2]-z)*pix_spacing[2];
    r = sqrt( dx*dx + dy*dy + dz*dz);
    r = r / radius;

 //   if (r>1) return 0.;
 //   val = (1-r)*(1-r)*(1-r)*(1-r)*(4*r+1.); // Wendland
	val = exp( -r*r );   
	return val;
}

/*
LF + u(LF) + alpha*rbf(LF,LF) = LM  (one landmark, 1D)
Solve for alpha; LF=fixed landmark, LM=moving landmark,
LW=warped landmark (moving displaced by u(x)).
RBFs are centered on fixed landmarks
*/
float bspline_rbf_score_distanceonly (
    float *trial_rbf_coeff,  
    int num_rbf_coeff, 
    void *score_data
)
{
    Rbf_parms *rbf_par = (Rbf_parms *) score_data;
    Bspline_parms *parms = rbf_par->bparms;
    Bspline_landmarks *blm;
    float rbfv, ds,score=0;
    int i,j,d,d1, rbfcenter[3];

    blm = parms->landmarks;

    for(i=0;i<blm->num_landmarks;i++) {
	for(d1=0;d1<3;d1++) {	
	    ds = blm->fixed_landmarks->points[3*i+d1] 
		+ blm->landmark_dxyz[3*i+d1] 
		- blm->moving_landmarks->points[3*i+d1];

	    for(j=0;j<blm->num_landmarks;j++) {
		//where are the centers?
		//for(d=0;d<3;d++) rbfcenter[d] = blm->landvox_warp[3*j+d]; 
		for(d=0;d<3;d++) rbfcenter[d] = blm->landvox_fix[3*j+d];

		rbfv = rbf_value( rbfcenter, blm->landvox_fix[3*i+0],
		    blm->landvox_fix[3*i+1],
		    blm->landvox_fix[3*i+2], 
		    rbf_par->radius,
		    rbf_par->vector_field->pix_spacing );

		ds = ds + trial_rbf_coeff[3*j+d1]* rbfv ;
	    } // end for j

	    score = score + ds*ds;
	} //end for d1
    }// end for i

    return score;
}


/*
Analytic expression for the integral of squared second derivatives
of the vector field of a single Gaussian RBF of radius c
*/
static float rbf_gauss_secderiv_self(float c)
{
float factor = 1.968701243; // pow( M_PI/2, 3./2);
return 15*factor/c;
} 
  
/*
Analytic expression for the integral of squared second derivatives
of the vector field of two Gaussian RBFs of radii c,
separated by squared distance a2 = (x1-x2)*(x1-x2)+...
(one-half of the overlap integral only).
*/
static float rbf_gauss_secderiv_cross(float c, float a2)
{
float factor = 1.968701243; // pow( M_PI/2, 3./2);
return factor/c*exp(-a2/(2*c*c))*(-10. + ( a2/(c*c)-5.)*(a2/(c*c)-5.) );
}

static float bspline_rbf_analytic_integral( Bspline_parms *parms, float *pix_spacing )
{
Bspline_landmarks *blm = parms->landmarks;
int i,j,d;
float a2, s = 0.;

for(i=0;i<blm->num_landmarks;i++)
for(d=0;d<3;d++)
	s += blm->rbf_coeff[3*i+d] * blm->rbf_coeff[3*i+d] *
		rbf_gauss_secderiv_self( parms->rbf_radius );

if (blm->num_landmarks>1)
for(i=0;i<blm->num_landmarks;i++)
for(j=i+1;j<blm->num_landmarks;j++)
for(d=0;d<3;d++)
{
		a2 = (blm->landvox_fix[3*i+0]-blm->landvox_fix[3*j+0])*
					 (blm->landvox_fix[3*i+0]-blm->landvox_fix[3*j+0])*
					 pix_spacing[0] * pix_spacing[0] +
					 (blm->landvox_fix[3*i+1]-blm->landvox_fix[3*j+1])*
					 (blm->landvox_fix[3*i+1]-blm->landvox_fix[3*j+1])*
					 pix_spacing[1] * pix_spacing[1] +
					 (blm->landvox_fix[3*i+2]-blm->landvox_fix[3*j+2])*
					 (blm->landvox_fix[3*i+2]-blm->landvox_fix[3*j+2])*
					  pix_spacing[2] * pix_spacing[2];
	
		s += 2. * blm->rbf_coeff[3*i+d] * blm->rbf_coeff[3*j+d] *
		rbf_gauss_secderiv_cross( parms->rbf_radius, a2 );
}

//printf("Analytic INTSECDER: %g\n", s);
return s;
}

static float bspline_rbf_analytic_integral_detailed( Bspline_parms *parms, float *pix_spacing )
{
Bspline_landmarks *blm = parms->landmarks;
int i,j,d;
float a2, s = 0.;
float t;

t = rbf_gauss_secderiv_self( parms->rbf_radius);
printf("Self-integral: %f\n",t);

for(i=0;i<blm->num_landmarks;i++)
for(d=0;d<3;d++)
	s += blm->rbf_coeff[3*i+d] * blm->rbf_coeff[3*i+d] *
		rbf_gauss_secderiv_self( parms->rbf_radius );

if (blm->num_landmarks>1)
for(i=0;i<blm->num_landmarks;i++)
for(j=i+1;j<blm->num_landmarks;j++)
for(d=0;d<3;d++)
{
		a2 = (blm->landvox_fix[3*i+0]-blm->landvox_fix[3*j+0])*
					 (blm->landvox_fix[3*i+0]-blm->landvox_fix[3*j+0])*
					 pix_spacing[0] * pix_spacing[0] +
					 (blm->landvox_fix[3*i+1]-blm->landvox_fix[3*j+1])*
					 (blm->landvox_fix[3*i+1]-blm->landvox_fix[3*j+1])*
					 pix_spacing[1] * pix_spacing[1] +
					 (blm->landvox_fix[3*i+2]-blm->landvox_fix[3*j+2])*
					 (blm->landvox_fix[3*i+2]-blm->landvox_fix[3*j+2])*
					  pix_spacing[2] * pix_spacing[2];
	
		s += 2. * blm->rbf_coeff[3*i+d] * blm->rbf_coeff[3*j+d] *
		rbf_gauss_secderiv_cross( parms->rbf_radius, a2 );

		t = rbf_gauss_secderiv_cross( parms->rbf_radius, a2);
		printf("Cross-int: %f\n", t);

}

//printf("Analytic INTSECDER: %g\n", s);
return s;
}


/*
LF + u(LF) + alpha*rbf(LF,LF) = LM  (one landmark, 1D)
WITH REGULARIZATION
Solve for alpha; LF=fixed landmark, LM=moving landmark,
LW=warped landmark (moving displaced by u(x)).
RBFs are centered on fixed landmarks
*/
float bspline_rbf_score (
    float *trial_rbf_coeff,  
    int num_rbf_coeff, 
    void *score_data
)
{
    Rbf_parms *rbf_par = (Rbf_parms *) score_data;
    Bspline_parms *parms = rbf_par->bparms;
    Bspline_landmarks *blm;
    float rbfv, ds,score=0;
    int i,j,d,d1, rbfcenter[3];

    blm = parms->landmarks;

    for(i=0;i<blm->num_landmarks;i++) {
	for(d1=0;d1<3;d1++) {	
	    ds = blm->fixed_landmarks->points[3*i+d1] 
		+ blm->landmark_dxyz[3*i+d1] 
		- blm->moving_landmarks->points[3*i+d1];

	    for(j=0;j<blm->num_landmarks;j++) {
		//where are the centers?
		//for(d=0;d<3;d++) rbfcenter[d] = blm->landvox_warp[3*j+d]; 
		for(d=0;d<3;d++) rbfcenter[d] = blm->landvox_fix[3*j+d];

		rbfv = rbf_value( rbfcenter, blm->landvox_fix[3*i+0],
		    blm->landvox_fix[3*i+1],
		    blm->landvox_fix[3*i+2], 
		    rbf_par->radius,
		    rbf_par->vector_field->pix_spacing );

		ds = ds + trial_rbf_coeff[3*j+d1]* rbfv ;
	    } // end for j

	    score = score + ds*ds;
	} //end for d1
    }// end for i

// adding second derivatives cost
	
	for(i=0;i<3*blm->num_landmarks;i++)
		blm->rbf_coeff[i] = trial_rbf_coeff[i]; // for analytic integral

	score = score + parms->rbf_young_modulus*bspline_rbf_analytic_integral( parms, rbf_par->vector_field->pix_spacing );
	
	return score;
}


float bspline_rbf_score_detailed (
    float *trial_rbf_coeff,  
    int num_rbf_coeff, 
    void *score_data
)
{
    Rbf_parms *rbf_par = (Rbf_parms *) score_data;
    Bspline_parms *parms = rbf_par->bparms;
    Bspline_landmarks *blm;
    float rbfv, ds,score=0;
    int i,j,d,d1, rbfcenter[3];
	float t;

    blm = parms->landmarks;

    for(i=0;i<blm->num_landmarks;i++) {
	for(d1=0;d1<3;d1++) {	
	    ds = blm->fixed_landmarks->points[3*i+d1] 
		+ blm->landmark_dxyz[3*i+d1] 
		- blm->moving_landmarks->points[3*i+d1];

		printf("i %d d %d rhs = %f\n",i, d1, ds);

	    for(j=0;j<blm->num_landmarks;j++) {
		//where are the centers?
		//for(d=0;d<3;d++) rbfcenter[d] = blm->landvox_warp[3*j+d]; 
		for(d=0;d<3;d++) rbfcenter[d] = blm->landvox_fix[3*j+d];

		rbfv = rbf_value( rbfcenter, blm->landvox_fix[3*i+0],
		    blm->landvox_fix[3*i+1],
		    blm->landvox_fix[3*i+2], 
		    rbf_par->radius,
		    rbf_par->vector_field->pix_spacing );

		ds = ds + trial_rbf_coeff[3*j+d1]* rbfv ;
	    
		printf("i %d j %d d %d  rbfv %f  coeff %f\n", i,j,d1, rbfv, trial_rbf_coeff[3*j+d1]);
		
		} // end for j

		printf("ds %f\n", ds);

	    score = score + ds*ds;
	} //end for d1
    }// end for i

printf("dist score %f\n", score);

// adding second derivatives cost
	
	for(i=0;i<3*blm->num_landmarks;i++)
		blm->rbf_coeff[i] = trial_rbf_coeff[i]; // for analytic integral

	t = parms->rbf_young_modulus*bspline_rbf_analytic_integral( parms, rbf_par->vector_field->pix_spacing );

	printf("Sec der score: %f\n", t);

	score = score + t; 
	
	printf("score = %f\n", score);
	return score;
}

/*
Test RBF solution by calculating the residual error 
of landmark matching 
*/
static void bspline_rbf_test_solution(Volume *vector_field, Bspline_parms *parms)
{
Bspline_landmarks *blm = parms->landmarks;
float *vf;
float totdist, dx, dist, maxdist; 
int i, j, d, dd, fv, rbfcenter[3];

Rbf_parms *rbf_par;

rbf_par = (Rbf_parms *)malloc( sizeof(Rbf_parms));
rbf_par->radius = parms->rbf_radius;
rbf_par->bparms = parms;
rbf_par->vector_field = vector_field;

    if (vector_field->pix_type != PT_VF_FLOAT_INTERLEAVED )
	print_and_exit("Sorry, this type of vector field is not supported\n");

    //fill in vector field at fixed landmark location
    vf = (float*) vector_field->img;

    for(i=0;i<blm->num_landmarks;i++) {
	fv = blm->landvox_fix[3*i+2] * vector_field->dim[0] * vector_field->dim[1] 
	    +blm->landvox_fix[3*i+1] * vector_field->dim[0] 
	    +blm->landvox_fix[3*i+0] ;

	for(d=0;d<3;d++) blm->landmark_dxyz[3*i+d] = vf[3*fv+d];
    }

maxdist = -1;
totdist = 0;
for(i=0;i<blm->num_landmarks;i++)
{
dist = 0;
for(d = 0; d<3; d++)
{
	dx = blm->fixed_landmarks->points[3*i+d] 
		    + blm->landmark_dxyz[3*i+d] 
			- blm->moving_landmarks->points[3*i+d];
for(j=0;j<blm->num_landmarks;j++)
{

    for(dd=0;dd<3;dd++) rbfcenter[dd] = blm->landvox_fix[3*j+dd];
	dx += blm->rbf_coeff[3*j+d]* rbf_value( rbfcenter, 
				blm->landvox_fix[3*i+0],
				blm->landvox_fix[3*i+1],
				blm->landvox_fix[3*i+2], 
				rbf_par->radius,
				rbf_par->vector_field->pix_spacing );
					
}
dist = dist + dx*dx;
}
dist = sqrt(dist);
totdist += dist;
if (dist>maxdist) maxdist = dist;
//printf("L%2d  %f\n", i, dist);
}

totdist/=blm->num_landmarks;

printf("Landmark mismatch: AVERAGE %f, MAX %f\n", totdist, maxdist);
}


// find RBF coeff by solving the linear equations using ITK's SVD routine
// using analytical regularization of Gaussian exp(-r*r) RBFs to minimize
// the sum of squares of second derivatives.
// Output:
// parms->blm->rbf_coeff contains RBF coefficients
void bspline_rbf_find_coeffs_reg(Volume *vector_field, Bspline_parms *parms)
{
    Bspline_landmarks *blm = parms->landmarks;
    float *vf;
    float rbfv1, rbfv2;
    int i, j, k, d, fv, rbfcenter[3];

	//begin regularization
	float rbf_young_modulus = parms->rbf_young_modulus;
	float rbf_prefactor, reg_term, r2, tmp;
	int d1;
	//end regularization

    typedef vnl_matrix <double> Vnl_matrix;
    typedef vnl_svd <double> SVDSolverType;
    Vnl_matrix A, b;

    Rbf_parms *rbf_par;

    rbf_par = (Rbf_parms *)malloc( sizeof(Rbf_parms));
    rbf_par->radius = parms->rbf_radius;
    rbf_par->bparms = parms;
    rbf_par->vector_field = vector_field;

	printf("Finding RBF coeffs, radius %.2f, Young modulus %e\n", parms->rbf_radius, parms->rbf_young_modulus);

    if (vector_field->pix_type != PT_VF_FLOAT_INTERLEAVED )
	print_and_exit("Sorry, this type of vector field is not supported\n");

    //fill in vector field at fixed landmark location
    vf = (float*) vector_field->img;

    for(i=0;i<blm->num_landmarks;i++) {
	fv = blm->landvox_fix[3*i+2] * vector_field->dim[0] * vector_field->dim[1] 
	    +blm->landvox_fix[3*i+1] * vector_field->dim[0] 
	    +blm->landvox_fix[3*i+0] ;

	for(d=0;d<3;d++) blm->landmark_dxyz[3*i+d] = vf[3*fv+d];
    }

    A.set_size (3 * blm->num_landmarks, 3 * blm->num_landmarks);
//    A.set_identity ();
	A.fill(0.);

    b.set_size (3 * blm->num_landmarks, 1);
    b.fill (0.0);

    // right-hand side
	
	for(i=0;i<blm->num_landmarks;i++) {
	for(j=0;j<blm->num_landmarks;j++) {
	
    for(d=0;d<3;d++) rbfcenter[d] = blm->landvox_fix[3*i+d];
	
	    rbfv1 = rbf_value( rbfcenter, 
		blm->landvox_fix[3*j+0],
		blm->landvox_fix[3*j+1],
		blm->landvox_fix[3*j+2], 
		rbf_par->radius,
		rbf_par->vector_field->pix_spacing );
		
	for(d=0;d<3;d++) {
	    b (3*i +d, 0) -= rbfv1* (blm->fixed_landmarks->points[3*j+d] 
		    + blm->landmark_dxyz[3*j+d] 
		    - blm->moving_landmarks->points[3*j+d]);
	}
    }
	}

    // matrix
    for(i=0;i<blm->num_landmarks;i++) {
	for(j=0;j<blm->num_landmarks;j++) {

		tmp = 0;
		for(k=0;k<blm->num_landmarks;k++) {

	    for(d=0;d<3;d++) rbfcenter[d] = blm->landvox_fix[3*k+d];
	
	    rbfv1 = rbf_value( rbfcenter, 
		blm->landvox_fix[3*i+0],
		blm->landvox_fix[3*i+1],
		blm->landvox_fix[3*i+2], 
		rbf_par->radius,
		rbf_par->vector_field->pix_spacing );

		rbfv2 = rbf_value( rbfcenter, 
		blm->landvox_fix[3*j+0],
		blm->landvox_fix[3*j+1],
		blm->landvox_fix[3*j+2], 
		rbf_par->radius,
		rbf_par->vector_field->pix_spacing );

		tmp += rbfv1*rbfv2;
		}

	    for(d=0;d<3;d++)
		{
		A(3*i+d, 3*j+d) = tmp ;
		//printf("i j d = %d %d %d   A = %f\n", i,j,d, A(3*i+d, 3*j+d));
		}

	}
    }

	//begin regularization
	//add extra terms to the matrix
	rbf_prefactor = sqrt(M_PI/2.)*sqrt(M_PI/2.)*sqrt(M_PI/2.)/rbf_par->radius;
	for(d=0;d<3;d++)
	{
		for(i=0;i<blm->num_landmarks;i++) 
		{
			for(j=0;j<blm->num_landmarks;j++)
			{

			tmp = A(3*i+d, 3*j+d);
			reg_term = 0.;			
			
			if (i==j) { reg_term = rbf_prefactor * 15.; }
			else
			{
				// distance between landmarks i,j in mm
				r2 = (blm->landvox_fix[3*i+0]-blm->landvox_fix[3*j+0])*
					 (blm->landvox_fix[3*i+0]-blm->landvox_fix[3*j+0])*
					 rbf_par->vector_field->pix_spacing[0]*
					 rbf_par->vector_field->pix_spacing[0] +
					 (blm->landvox_fix[3*i+1]-blm->landvox_fix[3*j+1])*
					 (blm->landvox_fix[3*i+1]-blm->landvox_fix[3*j+1])*
					 rbf_par->vector_field->pix_spacing[1]*
					 rbf_par->vector_field->pix_spacing[1] +
					 (blm->landvox_fix[3*i+2]-blm->landvox_fix[3*j+2])*
					 (blm->landvox_fix[3*i+2]-blm->landvox_fix[3*j+2])*
					 rbf_par->vector_field->pix_spacing[2]*
					 rbf_par->vector_field->pix_spacing[2];

				r2 = r2 / (rbf_par->radius * rbf_par->radius );
				reg_term = rbf_prefactor * exp(-r2/2.) * (-10 + (r2-5.)*(r2-5.));
			}
		//	printf("i j d = %d %d %d  regterm = %f\n", i,j,d, reg_term);
			A(3*i+d,3*j+d) = tmp + reg_term * rbf_young_modulus;
			}
		}
	}
	//end regularization

//    A.print (std::cout);
//    b.print (std::cout);

    SVDSolverType svd (A, 1e-6);
    Vnl_matrix x = svd.solve (b);

//    x.print (std::cout);

    for(i=0;i<3*blm->num_landmarks;i++) blm->rbf_coeff[i] = x(i,0);

//	bspline_rbf_analytic_integral( parms, rbf_par->vector_field->pix_spacing );
	bspline_rbf_test_solution( vector_field, parms);

/* checking the matrix solution
	float dx, totdx = 0;
	for(i=0;i<3*blm->num_landmarks;i++)
	{
	dx = (blm->fixed_landmarks->points[i] 
		    + blm->landmark_dxyz[i] 
		    - blm->moving_landmarks->points[i]);
	for(j=0;j<3*blm->num_landmarks;j++)
		dx += A(i,j)*blm->rbf_coeff[j];
	totdx += dx*dx;
	}
	totdx = sqrt(totdx)/(3*blm->num_landmarks);
	printf("SVD residual error %f\n", totdx); */

//    printf("rbf coeff from optimize reg:  %.3f  %.3f  %.3f   dist unkn\n", 
//	blm->rbf_coeff[0], blm->rbf_coeff[1], blm->rbf_coeff[2] );


}

// find RBF coeff by solving the linear equations using ITK's SVD routine
// Output:
// parms->blm->rbf_coeff contains RBF coefficients
void bspline_rbf_find_coeffs_noreg(Volume *vector_field, Bspline_parms *parms)
{
    Bspline_landmarks *blm = parms->landmarks;
    float *vf;
    float rbfv;
    int i, j, d, fv, rbfcenter[3];

    typedef vnl_matrix <double> Vnl_matrix;
    typedef vnl_svd <double> SVDSolverType;
    Vnl_matrix A, b;

    Rbf_parms *rbf_par;

    rbf_par = (Rbf_parms *)malloc( sizeof(Rbf_parms));
    rbf_par->radius = parms->rbf_radius;
    rbf_par->bparms = parms;
    rbf_par->vector_field = vector_field;

    if (vector_field->pix_type != PT_VF_FLOAT_INTERLEAVED )
	print_and_exit("Sorry, this type of vector field is not supported\n");

    //fill in vector field at fixed landmark location
    vf = (float*) vector_field->img;

    for(i=0;i<blm->num_landmarks;i++) {
	fv = blm->landvox_fix[3*i+2] * vector_field->dim[0] * vector_field->dim[1] 
	    +blm->landvox_fix[3*i+1] * vector_field->dim[0] 
	    +blm->landvox_fix[3*i+0] ;

	for(d=0;d<3;d++) blm->landmark_dxyz[3*i+d] = vf[3*fv+d];
    }

    A.set_size (3 * blm->num_landmarks, 3 * blm->num_landmarks);
    A.set_identity ();

    b.set_size (3 * blm->num_landmarks, 1);
    b.fill (0.0);

    // right-hand side
    for(i=0;i<blm->num_landmarks;i++) {
	for(d=0;d<3;d++) {
	    b (3*i +d, 0) = 
		-(blm->fixed_landmarks->points[3*i+d] 
		    + blm->landmark_dxyz[3*i+d] 
		    - blm->moving_landmarks->points[3*i+d]);
	}
    }

    // matrix
    for(i=0;i<blm->num_landmarks;i++) {
	for(j=0;j<blm->num_landmarks;j++) {

	    for(d=0;d<3;d++) rbfcenter[d] = blm->landvox_fix[3*j+d];
	
	    rbfv = rbf_value( rbfcenter, 
		blm->landvox_fix[3*i+0],
		blm->landvox_fix[3*i+1],
		blm->landvox_fix[3*i+2], 
		rbf_par->radius,
		rbf_par->vector_field->pix_spacing );

	    for(d=0;d<3;d++)
		A(3*i+d, 3*j+d) = rbfv ;

	}
    }

    A.print (std::cout);
    b.print (std::cout);

    SVDSolverType svd (A, 1e-6);
    Vnl_matrix x = svd.solve (b);

    x.print (std::cout);

    for(i=0;i<3*blm->num_landmarks;i++) blm->rbf_coeff[i] = x(i,0);

    printf("rbf coeff from optimize:  %.3f  %.3f  %.3f   dist unkn\n", 
	blm->rbf_coeff[0], blm->rbf_coeff[1], blm->rbf_coeff[2] );

}


/*
Optimization procedure is used even if no regularization required,
to have a single data pathway
Output:
parms->blm->rbf_coeff contains RBF coefficients
*/
void bspline_rbf_find_coeffs_NM( Volume *vector_field, Bspline_parms *parms )
{
    Bspline_landmarks *blm = parms->landmarks;
    float *rbf_simplex, *scores, *vf;
    int i,d,fv;
    Rbf_parms *rbf_par;

    rbf_par = (Rbf_parms *)malloc( sizeof(Rbf_parms));
    rbf_par->radius = parms->rbf_radius;
    rbf_par->bparms = parms;
    rbf_par->vector_field = vector_field;

    if (vector_field->pix_type != PT_VF_FLOAT_INTERLEAVED )
	print_and_exit("Sorry, this type of vector field is not supported\n");

    rbf_simplex = (float *)malloc(3*blm->num_landmarks*3*(blm->num_landmarks+1)*sizeof(float));
    scores = (float *)malloc(3*(blm->num_landmarks+1)*sizeof(float));

    //fill in vector field at fixed landmark location
    vf = (float*) vector_field->img;

    for(i=0;i<blm->num_landmarks;i++) {
	fv = blm->landvox_fix[3*i+2] * vector_field->dim[0] * vector_field->dim[1] 
	    +blm->landvox_fix[3*i+1] * vector_field->dim[0] 
	    +blm->landvox_fix[3*i+0] ;

	for(d=0;d<3;d++) blm->landmark_dxyz[3*i+d] = vf[3*fv+d];
    }

    //initial random simplex, plus-minus 10 mm displacement of each landmark
    for(i=0; i<3*blm->num_landmarks*3*(blm->num_landmarks+1); i++ )
	rbf_simplex[i] = -10.+20.*(float)rand()/RAND_MAX;

    minimize_nelder_mead( rbf_simplex, scores, 3*blm->num_landmarks, bspline_rbf_score, rbf_par);

    //copy first vertex to blm->rbf_coeffs, as output
    for(i=0;i<3*blm->num_landmarks;i++) blm->rbf_coeff[i] = rbf_simplex[i];

//    printf("rbf coeff from optimize NM:   %.3f  %.3f  %.3f   dist %.3f\n", 
//	blm->rbf_coeff[0], blm->rbf_coeff[1], blm->rbf_coeff[2],
//	scores[0]
//    );

//	bspline_rbf_score_detailed(blm->rbf_coeff, 3*blm->num_landmarks, rbf_par);

    free(rbf_simplex);
    free(scores);
    free(rbf_par);
}

void bspline_rbf_find_coeffs(Volume *vector_field, Bspline_parms *parms)
{
float s;
int i,d;

bspline_rbf_find_coeffs_reg(vector_field, parms);

for(i=0;i<parms->landmarks->num_landmarks;i++)
printf("coeff SVD %d   %.4f %.4f %.4f\n",  i,
	   parms->landmarks->rbf_coeff[3*i+0],
	   parms->landmarks->rbf_coeff[3*i+1],parms->landmarks->rbf_coeff[3*i+2]);

/*
for(d=0;d<3;d++)
for(i=0;i<parms->landmarks->num_landmarks;i++)
parms->landmarks->rbf_coeff[3*i+d] = 0;
bspline_rbf_find_coeffs_NM(vector_field, parms);
for(i=0;i<parms->landmarks->num_landmarks;i++)
printf("coeff NM  %d   %.4f %.4f %.4f\n",  i,
	   parms->landmarks->rbf_coeff[3*i+0],
	   parms->landmarks->rbf_coeff[3*i+1],parms->landmarks->rbf_coeff[3*i+2]);
*/

//s = bspline_rbf_analytic_integral_detailed( parms, vector_field->pix_spacing );
s = bspline_rbf_analytic_integral( parms, vector_field->pix_spacing );
printf("Analytic int sec der %f\n",s);

}

/*
Adds RBF contributions to the vector field
landmark_dxyz is not updated by this function
Version for RBF with local support
*/
void
bspline_rbf_update_vector_field_truncated (
    Volume *vector_field,
    Bspline_parms *parms 
)
{
    Bspline_landmarks *blm = parms->landmarks;
    int lidx, d, fv;
    int fi, fj, fk;
    int *rbf_vox_origin;
    float *vf;
    float rbf;
    float dr;

    printf("RBF, updating the vector field\n");

    if (vector_field->pix_type != PT_VF_FLOAT_INTERLEAVED )
	print_and_exit("Sorry, this type of vector field is not supported\n");

    dr = 3*parms->rbf_radius+1;
    vf = (float*) vector_field->img;
    //where are the centers?
    //rbf_vox_origin = blm->landvox_warp;
    rbf_vox_origin = blm->landvox_fix;

    //RBF contributions added to vector field
    for (lidx=0; lidx < blm->num_landmarks; lidx++) {
	for (fk = rbf_vox_origin[2+3*lidx] - dr; fk < rbf_vox_origin[2+3*lidx] + dr; fk++) {
	    for (fj = rbf_vox_origin[1+3*lidx] - dr; fj < rbf_vox_origin[1+3*lidx] + dr; fj++) {
		for (fi = rbf_vox_origin[0+3*lidx] - dr; fi < rbf_vox_origin[0+3*lidx] + dr; fi++) {
		    if (fi < 0 || fi >= vector_field->dim[0]) continue;
		    if (fj < 0 || fj >= vector_field->dim[1]) continue;
		    if (fk < 0 || fk >= vector_field->dim[2]) continue;

		    fv = fk * vector_field->dim[0] * vector_field->dim[1] 
			+ fj * vector_field->dim[0] + fi;
			
		    rbf = rbf_value (rbf_vox_origin+3*lidx,  
			fi, fj, fk, 
			parms->rbf_radius, 
			vector_field->pix_spacing);

		    for(d=0;d<3;d++) {
			vf[3*fv+d] += blm->rbf_coeff[3*lidx+d]* rbf;
#if defined (commentout)
			printf ("Adding: %d (%d %d %d) (%g * %g) %g\n", 
			    lidx, 
			    fi, fj, fk, 
			    blm->rbf_coeff[3*lidx+d], rbf, 
			    blm->rbf_coeff[3*lidx+d] * rbf);
#endif
		    }
		}
	    }
	}
    }
}

/*
Adds RBF contributions to the vector field
landmark_dxyz is not updated by this function
Version without truncation: scan over the entire vf
and add up all RBFs in each voxel
*/
void
bspline_rbf_update_vector_field (
    Volume *vector_field,
    Bspline_parms *parms 
)
{
    Bspline_landmarks *blm = parms->landmarks;
    int lidx, d, fv;
    int fi, fj, fk;
    int *rbf_vox_origin;
    float *vf;
    float rbf;

    printf("RBF, updating the vector field\n");

    if (vector_field->pix_type != PT_VF_FLOAT_INTERLEAVED )
	print_and_exit("Sorry, this type of vector field is not supported\n");

    vf = (float*) vector_field->img;
	rbf_vox_origin = blm->landvox_fix;

    //RBF contributions added to vector field

    for ( fk = 0; fk < vector_field->dim[2];  fk++) {
	for ( fj = 0; fj < vector_field->dim[1];  fj++) {
	for ( fi = 0; fi < vector_field->dim[0];  fi++) {
		
		for (lidx=0; lidx < blm->num_landmarks; lidx++) {

		    fv = fk * vector_field->dim[0] * vector_field->dim[1] 
			+ fj * vector_field->dim[0] + fi;
			
		    rbf = rbf_value (rbf_vox_origin+3*lidx,  
			fi, fj, fk, 
			parms->rbf_radius, 
			vector_field->pix_spacing);

		    for(d=0;d<3;d++) {
			vf[3*fv+d] += blm->rbf_coeff[3*lidx+d]* rbf;
#if defined (commentout)
			printf ("Adding: %d (%d %d %d) (%g * %g) %g\n", 
			    lidx, 
			    fi, fj, fk, 
			    blm->rbf_coeff[3*lidx+d], rbf, 
			    blm->rbf_coeff[3*lidx+d] * rbf);
#endif
		    }
		}
	    }
	}
    }
}
