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

typedef struct rbf_params Rbf_parms;
struct rbf_params { // used to pass information to bspline_rbf_score
float radius;    // radius of the RBF
BSPLINE_Parms *bparms;
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
void minimize_nelder_mead(float *x, float *funcval, int n, float(*func)(float *, int, void *), void *params )
{
int iter=0;
int i,j;
float *x0, *xr, *xe, *xc;
float val_xr, val_xe, val_xc;
float alpha=1., gamma=2., rho=0.5, sigma=0.5;

x0 = (float *)malloc( n*sizeof(int) );
xr = (float *)malloc( n*sizeof(int) );
xe = (float *)malloc( n*sizeof(int) );
xc = (float *)malloc( n*sizeof(int) );

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
float rbf_value( int *center, int x, int y, int z, float radius, float *pix_spacing )
{
float val, r, dx,dy,dz;

dx = (center[0]-x)*pix_spacing[0];
dy = (center[1]-y)*pix_spacing[1];
dz = (center[2]-z)*pix_spacing[2];
r = sqrt( dx*dx + dy*dy + dz*dz);
r = r / radius;

if (r>1) return 0.;
val = (1-r)*(1-r)*(1-r)*(1-r)*(4*r+1.); // Wendland
return val;
}


/*
LF + u(LF) + alpha*rbf(LF,LW) = LM  (one landmark, 1D)
Solve for alpha; LF=fixed landmark, LM=moving landmark,
LW=warped landmark (moving displaced by u(x)).
RBFs are centered on warped landmarks
*/
float bspline_rbf_score( float *trial_rbf_coeff,  int num_rbf_coeff, Rbf_parms *rbf_par)
{
BSPLINE_Parms *parms = rbf_par->bparms;
Bspline_landmarks *blm;
float rbfv, ds,score=0;
int i,j,d,d1, rbfcenter[3];

blm = parms->landmarks;

for(i=0;i<blm->num_landmarks;i++) {
	for(d1=0;d1<3;d1++) {	
	ds = blm->fixed_landmarks[3*i+d1] + blm->landmark_dxyz[3*i+d1] - blm->moving_landmarks[3*i+d1];

	for(j=0;j<blm->num_landmarks;j++) {
		for(d=0;d<3;d++) rbfcenter[d] = blm->landvox_warp[3*j+d]; 

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
Optimization procedure is used even if no regularization required,
to have a single data pathway
Output:
parms->blm->rbf_coeff contains RBF coefficients
*/
void bspline_rbf_find_coeffs( Volume *vector_field, BSPLINE_Parms *parms )
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
output:
parms->landmarks->warped_landmarks contains landmark coordinates updated by RBF
parms->landmarks->landvox_warp: same, in voxels
Must be called AFTER bspline_rbf_update_vector_field

We must solve LW + u(LW) = LM to get new LW, corresponding to current vector field.
LW is not necessarily equal to LF, e.g. if regularization has been used
*/
void bspline_rbf_update_landmarks(
	Volume *vector_field, 
	BSPLINE_Parms *parms,
	BSPLINE_Xform* bxf, 
    Volume *fixed, 
    Volume *moving)
{
Bspline_landmarks *blm = parms->landmarks;
int ri, rj, rk;
int fi, fj, fk;
int mi, mj, mk;
float fx, fy, fz;
float mx, my, mz;
int i,d,fv, lidx;
float dd, *vf, dxyz[3], *dd_min;

dd_min = (float *)malloc( blm->num_landmarks * sizeof(float));
for(d=0;d<blm->num_landmarks;d++) dd_min[d] = 1e20F; //a very large number

if (vector_field->pix_type != PT_VF_FLOAT_INTERLEAVED )
		print_and_exit("Sorry, this type of vector field is not supported\n");	
	vf = (float *)vector_field->img;

	for (rk = 0, fk = bxf->roi_offset[2]; rk < bxf->roi_dim[2]; rk++, fk++) {
	fz = bxf->img_origin[2] + bxf->img_spacing[2] * fk;
	for (rj = 0, fj = bxf->roi_offset[1]; rj < bxf->roi_dim[1]; rj++, fj++) {
	    fy = bxf->img_origin[1] + bxf->img_spacing[1] * fj;
	    for (ri = 0, fi = bxf->roi_offset[0]; ri < bxf->roi_dim[0]; ri++, fi++) {
			fx = bxf->img_origin[0] + bxf->img_spacing[0] * fi;

			fv = fk * vector_field->dim[0] * vector_field->dim[1] + fj * vector_field->dim[0] +fi ;

			for(d=0;d<3;d++) dxyz[d] = vf[3*fv+d];

			/* Find correspondence in moving image */
			mx = fx + dxyz[0];
			mi = ROUND_INT ((mx - moving->offset[0]) / moving->pix_spacing[0]);
			if (mi < 0 || mi >= moving->dim[0]) continue;
			my = fy + dxyz[1];
			mj = ROUND_INT ((my - moving->offset[1]) / moving->pix_spacing[1]);
			if (mj < 0 || mj >= moving->dim[1]) continue;
			mz = fz + dxyz[2];
			mk = ROUND_INT ((mz - moving->offset[2]) / moving->pix_spacing[2]);
			if (mk < 0 || mk >= moving->dim[2]) continue;

			//saving vector field in a voxel which is the closest to landvox_mov
			//after being displaced by the vector field
			for( lidx = 0; lidx < blm->num_landmarks; lidx++) {
				dd = (mi - blm->landvox_mov[lidx*3+0]) * (mi - blm->landvox_mov[lidx*3+0])
					+(mj - blm->landvox_mov[lidx*3+1]) * (mj - blm->landvox_mov[lidx*3+1])
					+(mk - blm->landvox_mov[lidx*3+2]) * (mk - blm->landvox_mov[lidx*3+2]);
				if (dd < dd_min[lidx]) { 
									dd_min[lidx]=dd;   
									for(d=0;d<3;d++) blm->landmark_dxyz[3*lidx+d] = vf[3*fv+d];
									}
				} 
				}
			}
		}

for(i=0;i<blm->num_landmarks;i++)  {
	for(d=0; d<3; d++) 
	blm->warped_landmarks[3*i+d] = blm->moving_landmarks[3*i+d] - blm->landmark_dxyz[3*i+d];
	}

/* calculate voxel positions of warped landmarks  
 landmarks are stored internally without offsets.
 Offsets are used only when reading/writing landmarks form/to files
*/
	for (lidx = 0; lidx < blm->num_landmarks; lidx++) {
	for (d = 0; d < 3; d++) {
	    blm->landvox_warp[lidx*3 + d] 
		= ROUND_INT ((blm->warped_landmarks[lidx*3 + d] /*- fixed->offset[d]*/ )
		    / fixed->pix_spacing[d]);
	    if (blm->landvox_warp[lidx*3 + d] < 0 
		|| blm->landvox_warp[lidx*3 + d] >= fixed->dim[d])
	    {
		print_and_exit (
		    "Error, warped landmark %d outside of fixed image for dim %d.\n"
		    "Location in vox = %d\n"
		    "Image boundary in vox = (%d %d)\n",
		    lidx, d, blm->landvox_warp[lidx*3 + d], 0, fixed->dim[d]-1);
			}
		} 
	}


}

/*
Adds RBF contributions to the vector field
Must be called BEFORE bspline_rbf_update_landmarks,
as parms->landmarks->warped_landmarks and landvox_warp 
must contain values from a previous stage
landmark_dxyz is not updated by this function
*/
void
bspline_rbf_update_vector_field (
	Volume *vector_field,
	BSPLINE_Parms *parms 
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

	dr = parms->rbf_radius+1;
	vf = (float*) vector_field->img;
	rbf_vox_origin = blm->landvox_warp;

	//RBF contributions added to vector field
	for(lidx=0;lidx<blm->num_landmarks;lidx++) {
		for(fk = rbf_vox_origin[2+3*lidx] - dr; fk< rbf_vox_origin[2+3*lidx] + dr; fk++)
		for(fj = rbf_vox_origin[1+3*lidx] - dr; fj< rbf_vox_origin[1+3*lidx] + dr; fj++)
		for(fi = rbf_vox_origin[0+3*lidx] - dr; fi< rbf_vox_origin[0+3*lidx] + dr; fi++) {
			if (fi < 0 || fi >= vector_field->dim[0]) continue;
			if (fj < 0 || fj >= vector_field->dim[1]) continue;
			if (fk < 0 || fk >= vector_field->dim[2]) continue;

			fv = fk * vector_field->dim[0] * vector_field->dim[1] + fj * vector_field->dim[0] + fi ;
			
			rbf = rbf_value( rbf_vox_origin+3*lidx,  fi,fj,fk, parms->rbf_radius, vector_field->pix_spacing );  

			for(d=0;d<3;d++) vf[3*fv+d] += ( blm->rbf_coeff[3*lidx+d]* rbf );
			}
		}
}
