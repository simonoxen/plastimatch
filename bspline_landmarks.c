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
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"

Bspline_landmarks*
bspline_landmarks_create (void)
{
    Bspline_landmarks *blm;
    blm = (Bspline_landmarks*) malloc (sizeof (Bspline_landmarks));
    memset (blm, 0, sizeof (Bspline_landmarks));
    return blm;
}

void
bspline_landmarks_destroy (Bspline_landmarks* blm)
{
    if (blm->fixed_landmarks) {
	free (blm->fixed_landmarks);
    }
    if (blm->moving_landmarks) {
	free (blm->moving_landmarks);
    }
    if (blm->landvox_mov) {
	free (blm->landvox_mov);
    }
    if (blm->landvox_fix) {
	free (blm->landvox_fix);
    }
    if (blm->landvox_warp) {
	free (blm->landvox_warp);
    }
	if (blm->warped_landmarks) {
	free (blm->warped_landmarks);
	}
	if (blm->rbf_coeff) {
	free (blm->rbf_coeff);
	}
	if (blm->landmark_dxyz) {
	free (blm->landmark_dxyz);
	}
	free (blm);
}

static void
bspline_landmarks_load_file (float **landmarks, int *num_landmarks, char *fn)
{
    FILE *fp;

    fp = fopen (fn, "r");
    if (!fp) {
	print_and_exit ("cannot read landmarks from file: %s\n", fn);
    }

    *num_landmarks = 0;
    *landmarks = 0;
    while (!feof(fp)) {
	char s[1024];
	char *s2;
	float lm[3];
	int land_sel, land_vis;
	int rc;

        fgets (s, 1024, fp);
	if (feof(fp)) break;
        if (s[0]=='#') continue;

	// skip the label field assuming it does not contain commas
        s2 = strchr(s,',');
	
        rc = sscanf (s2, ",%f,%f,%f,%d,%d\n", 
	    &lm[0], &lm[1], &lm[2], &land_sel, &land_vis);
	if (rc != 5) {
	    print_and_exit ("Error parsing landmark file: %s\n", fn);
	}
	++(*num_landmarks);
	*landmarks = (float*) realloc (*landmarks, 
	    3 * (*num_landmarks) * sizeof(float));

	/* Note: Slicer landmarks are in RAS coordinates. 
	   Change RAS to LPS (i.e. ITK RAI). */
	(*landmarks)[((*num_landmarks)-1)*3 + 0] = - lm[0];
	(*landmarks)[((*num_landmarks)-1)*3 + 1] = - lm[1];
	(*landmarks)[((*num_landmarks)-1)*3 + 2] = lm[2];
    }
    fclose (fp);
}

void
bspline_landmarks_adjust (Bspline_landmarks *blm, Volume *fixed, Volume *moving)
{
    int i, d;

    // Save position of landmarks in voxels for vector field calculation
    // GCS: I think we only need fixed
	// NSh: No. Moving landmarks must be used for v.f. calculation
	// since it is the moving image which is warped after registration
    blm->landvox_mov = (int*) malloc (3 * blm->num_landmarks * sizeof(int));
    for (i = 0; i < blm->num_landmarks; i++) {
	for (d = 0; d < 3; d++) {
	    blm->landvox_mov[i*3 + d] 
		= ROUND_INT ((blm->moving_landmarks[i*3 + d] - moving->offset[d])
		    / moving->pix_spacing[d]);
	    if (blm->landvox_mov[i*3 + d] < 0 
		|| blm->landvox_mov[i*3 + d] >= moving->dim[d])
	    {
		print_and_exit (
		    "Error, landmark %d outside of moving image for dim %d.\n"
		    "Location in vox = %d\n"
		    "Image boundary in vox = (%d %d)\n",
		    i, d, blm->landvox_mov[i*3 + d], 0, moving->dim[d]-1);
	    }
	}
    }

	blm->landvox_fix = (int*) malloc (3 * blm->num_landmarks * sizeof(int));
    for (i = 0; i < blm->num_landmarks; i++) {
	for (d = 0; d < 3; d++) {
	    blm->landvox_fix[i*3 + d] 
		= ROUND_INT ((blm->fixed_landmarks[i*3 + d] - fixed->offset[d])
		    / fixed->pix_spacing[d]);
	    if (blm->landvox_fix[i*3 + d] < 0 
		|| blm->landvox_fix[i*3 + d] >= fixed->dim[d])
	    {
		print_and_exit (
		    "Error, landmark %d outside of fixed image for dim %d.\n"
		    "Location in vox = %d\n"
		    "Image boundary in vox = (%d %d)\n",
		    i, d, blm->landvox_fix[i*3 + d], 0, fixed->dim[d]-1);
	    }
	}
    }

	//re-basing landmarks to the origin of fixed image
    for (i = 0; i < blm->num_landmarks; i++) {
	for (d = 0; d < 3; d++) {
		blm->fixed_landmarks[i*3 + d] -= fixed->offset[d];
		blm->moving_landmarks[i*3 + d] -= moving->offset[d];
		}
	}
    printf ("Adjusting complete.\n");
}

Bspline_landmarks*
bspline_landmarks_load (char *fixed_fn, char *moving_fn)
{
    int num_fixed_landmarks, num_moving_landmarks;
    float *fixed_landmarks, *moving_landmarks;
    Bspline_landmarks *blm;

    bspline_landmarks_load_file (&fixed_landmarks, &num_fixed_landmarks, 
	fixed_fn);
    bspline_landmarks_load_file (&moving_landmarks, &num_moving_landmarks, 
	moving_fn);

    if (num_fixed_landmarks != num_moving_landmarks) {
	print_and_exit ("Error. Different number of landmarks in files: "
	    "%s (%d), %s (%d)\n", fixed_fn, num_fixed_landmarks, 
	    moving_fn, num_moving_landmarks);
    }
    
    blm = bspline_landmarks_create ();
    blm->num_landmarks = num_fixed_landmarks;
    blm->moving_landmarks = moving_landmarks;
    blm->fixed_landmarks = fixed_landmarks;

	blm->warped_landmarks = (float *)malloc( 3 * num_fixed_landmarks * sizeof(float));
	blm->landvox_warp = (int*) malloc (3 * blm->num_landmarks * sizeof(int));
	// just in case even if RBF are not used
	blm->rbf_coeff = (float *)malloc( 3 * num_fixed_landmarks * sizeof(float));
	blm->landmark_dxyz = (float *)malloc( 3 * num_fixed_landmarks * sizeof(float));

    printf ("Found %d landmark(s) from fixed to moving image\n", 
	blm->num_landmarks);

    return blm;
}

void bspline_landmarks_write_file( char *fn, char *title, float *coords, int n, float *offset)
{
FILE *fp;
int lidx;
fp = fopen(fn,"w");
    if (!fp) {
	print_and_exit ("cannot write landmarks to file: %s\n", fn);
    }
fprintf(fp,"# name = %s\n",title);

/* Changing LPS to RAS and shifting */
for(lidx=0;lidx<n;lidx++)
fprintf(fp, "W%d,%f,%f,%f,1,1\n", lidx,
		-offset[0]-coords[0+3*lidx], 
		-offset[1]-coords[1+3*lidx],  
		 offset[2]+coords[2+3*lidx] );

fclose(fp);
}

/* 
NSh Correct calculation of landmark score; major rewrite.
Based on bspline_score_a_mse iterator

Input: parms->landmark_stiffness = spring constant for attraction
between landmarks. landmark_stiffness = 0 corresponds to no constraints
on landmarks, so moving landmarks "follow" the moving image
as it is being deformed.

Returns: score

Other output: 
parms->landmarks->warped_landmarks are landmark positions (in mm) deformed
by the current vector field
For reading into Slicer, the warped_landmarks must be transformed back to RAS 
and fixed->offset must be added, see bspline_landmarks_write_file

Output files: warplist.fcsv is a fiducials file with landmarks 
on the warped image; distlist.dat are the distances between
corresponding landmarks in fixed and warped images.

Fixed image remains fixed throughout, so are its landmarks.
Moving image is warped at the very end of bspline_main,
after the vector field has been established. Thus moving
landmarks must change coordinates to remain "stuck" to their
corresponding features on the moving image, even if there
is no attraction.

LM = moving landmark (in voxels).
We must solve  x+u(x)=LM for x to find the vector field u(x) 
that must be used to update landmark positions. It is not u(LM)!
This is due to the definition of v.f. and warping in vf_warp()

Mar 30 2010 - NSh 
*/
void
bspline_landmarks_score (
    BSPLINE_Parms *parms, 
    Bspline_state *bst, 
    BSPLINE_Xform* bxf, 
    Volume *fixed, 
    Volume *moving
)
{
    BSPLINE_Score* ssd = &bst->ssd;
    Bspline_landmarks *blm = parms->landmarks;
    int lidx;
    FILE *fp, *fp2;
    float land_score, land_grad_coeff, land_rawdist;
	int ri, rj, rk;
    int fi, fj, fk;
    int mi, mj, mk;
    float fx, fy, fz;
    float mx, my, mz;
    int p[3];
    int q[3];
    float dxyz[3];
    int qidx;
	int d;
	float diff[3];
	float l_dist;
	float dc_dv[3];
	float dd, *dd_min;  //minimum distance between a displaced voxel in moving and landvox_mov
	int *land_p, *land_q;

	land_score = 0;
    land_rawdist = 0;
    land_grad_coeff = parms->landmark_stiffness / blm->num_landmarks;

	dd_min = (float *)malloc( blm->num_landmarks * sizeof(float));
	for(d=0;d<blm->num_landmarks;d++) dd_min[d] = 1e20F; //a very large number

	land_p = (int *)malloc( 3* blm->num_landmarks * sizeof(int));
	land_q = (int *)malloc( 3* blm->num_landmarks * sizeof(int));

//NSh: we must find the vector field acting on the voxels of MOVING image
//where moving landmarks sit, and apply this v.f. to the landmarks

	for (rk = 0, fk = bxf->roi_offset[2]; rk < bxf->roi_dim[2]; rk++, fk++) {
	p[2] = rk / bxf->vox_per_rgn[2];
	q[2] = rk % bxf->vox_per_rgn[2];
	fz = bxf->img_origin[2] + bxf->img_spacing[2] * fk;
	for (rj = 0, fj = bxf->roi_offset[1]; rj < bxf->roi_dim[1]; rj++, fj++) {
	    p[1] = rj / bxf->vox_per_rgn[1];
	    q[1] = rj % bxf->vox_per_rgn[1];
	    fy = bxf->img_origin[1] + bxf->img_spacing[1] * fj;
	    for (ri = 0, fi = bxf->roi_offset[0]; ri < bxf->roi_dim[0]; ri++, fi++) {
			p[0] = ri / bxf->vox_per_rgn[0];
			q[0] = ri % bxf->vox_per_rgn[0];
			fx = bxf->img_origin[0] + bxf->img_spacing[0] * fi;

			/* Get B-spline deformation vector */
			qidx = INDEX_OF (q, bxf->vox_per_rgn);
			bspline_interp_pix (dxyz, bxf, p, qidx);

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

			// Storing p and q for (mi,mj,mk) nearest to landvox_mov
			// Typically there is an exact match, but for large vector field gradients
			// a voxel can be missed. Also, if multiple voxels map into a landmark,
			// use just one of them to avoid multiple counting in the score.
			for( lidx = 0; lidx < blm->num_landmarks; lidx++) {
				dd = (mi - blm->landvox_mov[lidx*3+0]) * (mi - blm->landvox_mov[lidx*3+0])
					+(mj - blm->landvox_mov[lidx*3+1]) * (mj - blm->landvox_mov[lidx*3+1])
					+(mk - blm->landvox_mov[lidx*3+2]) * (mk - blm->landvox_mov[lidx*3+2]);
				if (dd < dd_min[lidx]) { 
									dd_min[lidx]=dd;   
									for (d = 0; d < 3; d++) { 
									land_p[lidx*3+d]=p[d];  
									land_q[lidx*3+d]=q[d]; 
									}
								} 
				}
			}
		}
	}


//displacing the landmarks and writing them out.
    fp  = fopen("warplist.fcsv","w");
    fp2 = fopen("distlist.dat","w");
    fprintf(fp,"# name = warped\n");

	for( lidx = 0; lidx < blm->num_landmarks; lidx++) {

		printf("at landmark %d: dd %.1f  dxyz  %.2f %.2f %.2f\n", lidx, dd_min[lidx], dxyz[0],dxyz[1],dxyz[2] );

		if (dd_min[lidx]>10) logfile_printf("Landmark WARNING: landmark far from nearest voxel\n");

		for (d = 0; d < 3; d++) p[d] = land_p[3*lidx+d];
		for (d = 0; d < 3; d++) q[d] = land_q[3*lidx+d];
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
		bspline_interp_pix (dxyz, bxf, p, qidx);

		//actually move the moving landmark; note the minus sign
		for (d = 0; d < 3; d++) blm->warped_landmarks[3*lidx+d] = blm->moving_landmarks[3*lidx + d] - dxyz[d];
		for (d = 0; d < 3; d++) diff[d] = blm->fixed_landmarks[3*lidx+d]-blm->warped_landmarks[3*lidx+d];
		l_dist = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
		land_score += l_dist;
		land_rawdist += sqrt(l_dist);
		for (d = 0; d < 3; d++) dc_dv[d] = -land_grad_coeff * diff[d]; 
		bspline_update_grad (bst, bxf, p, qidx, dc_dv);

		// Note: Slicer landmarks are in RAS coordinates. Change LPS to RAS 
				fprintf(fp, "W%d,%f,%f,%f,1,1\n", lidx,
								-fixed->offset[0]-blm->warped_landmarks[0+3*lidx], 
								-fixed->offset[1]-blm->warped_landmarks[1+3*lidx],  
								 fixed->offset[2]+blm->warped_landmarks[2+3*lidx] );
				fprintf(fp2,"W%d %.3f\n", lidx, sqrt(l_dist));
	}

    fclose(fp);
    fclose(fp2);

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



	free(dd_min);
	free(land_p);
	free(land_q);

	land_score = land_score * parms->landmark_stiffness / blm->num_landmarks;
    printf ("        LM DIST %.4f COST %.4f\n", land_rawdist, land_score);
    ssd->score += land_score;
}
