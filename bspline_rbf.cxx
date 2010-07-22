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

    for(iter = 0; iter<10000; iter++)
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
	    ds = blm->fixed_landmarks[3*i+d1] + blm->landmark_dxyz[3*i+d1] - blm->moving_landmarks[3*i+d1];

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

// find RBF coeff by solving the linear equations using ITK's SVD routine
// Output:
// parms->blm->rbf_coeff contains RBF coefficients
void bspline_rbf_find_coeffs(Volume *vector_field, Bspline_parms *parms)
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
    for(i=0;i<blm->num_landmarks;i++) 
		for(d=0;d<3;d++) 	
		    b(3*i +d, 0) = -(blm->fixed_landmarks[3*i+d] + blm->landmark_dxyz[3*i+d] - blm->moving_landmarks[3*i+d] );

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

    SVDSolverType svd (A, 1e-8);
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

    printf("rbf coeff from optimize:  %.3f  %.3f  %.3f   dist %.3f\n", 
	blm->rbf_coeff[0], blm->rbf_coeff[1], blm->rbf_coeff[2],
	scores[0]
    );

    free(rbf_simplex);
    free(scores);
    free(rbf_par);
}

/*
Adds RBF contributions to the vector field
landmark_dxyz is not updated by this function
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
