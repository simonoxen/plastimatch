/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "plmsys.h"

#include "bspline.h"
#include "bspline_landmarks.h"
#include "bspline_rbf.h"
#include "bspline_opts.h"
#include "plm_math.h"
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


// radial basis function
// center and x,y,z are in voxels
// radius is in mm
static float rbf_value (int *center, int x, int y, int z, 
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

return s;
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
static void bspline_rbf_find_coeffs_reg(Volume *vector_field, Bspline_parms *parms)
{
    Bspline_landmarks *blm = parms->landmarks;
    float *vf;
    float rbfv1, rbfv2;
    int i, j, k, d, fv, rbfcenter[3];
	float rbf_young_modulus = parms->rbf_young_modulus;
	float rbf_prefactor, reg_term, r2, tmp;

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

	//add regularization terms to the matrix
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

//    A.print (std::cout);
//    b.print (std::cout);

    SVDSolverType svd (A, 1e-6);
    Vnl_matrix x = svd.solve (b);

//    x.print (std::cout);

    for(i=0;i<3*blm->num_landmarks;i++) blm->rbf_coeff[i] = x(i,0);

#if defined (commentout)
//checking the matrix solution
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
	printf("SVD residual error %f\n", totdx);
#endif

}

// find RBF coeff by solving the linear equations using ITK's SVD routine
// Output:
// parms->blm->rbf_coeff contains RBF coefficients
static void 
bspline_rbf_find_coeffs_noreg (
    Volume *vector_field, 
    Bspline_parms *parms
)
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

//    A.print (std::cout);
//    b.print (std::cout);

    SVDSolverType svd (A, 1e-6);
    Vnl_matrix x = svd.solve (b);

//    x.print (std::cout);

    for(i=0;i<3*blm->num_landmarks;i++) blm->rbf_coeff[i] = x(i,0);
}


// find RBF coeff by solving the linear equations using ITK's SVD routine
// Output:
// parms->blm->rbf_coeff contains RBF coefficients
void bspline_rbf_find_coeffs(Volume *vector_field, Bspline_parms *parms)
{
float s;

bspline_rbf_find_coeffs_reg(vector_field, parms);
//bspline_rbf_find_coeffs_noreg(vector_field, parms);

#if defined (commentout)
int i;
for(i=0;i<parms->landmarks->num_landmarks;i++)
printf("coeff %d   %.4f %.4f %.4f\n",  i,
	   parms->landmarks->rbf_coeff[3*i+0],
	   parms->landmarks->rbf_coeff[3*i+1],
	   parms->landmarks->rbf_coeff[3*i+2]);
#endif

bspline_rbf_test_solution( vector_field, parms);
s = bspline_rbf_analytic_integral( parms, vector_field->pix_spacing );
printf("Analytic INTSECDER %f\n",s);

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
