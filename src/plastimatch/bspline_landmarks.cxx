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
#include "pointset.h"
#include "print_and_exit.h"
#include "volume_macros.h"

#if defined (commentout)
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
	pointset_destroy (blm->fixed_landmarks);
    }
    if (blm->moving_landmarks) {
	pointset_destroy (blm->moving_landmarks);
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
#endif

#if defined (commentout)
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
#endif

#if defined (commentout)
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
	    blm->landvox_mov[i*3 + d] = ROUND_INT (
		(blm->moving_landmarks->points[3*i + d]
		    - moving->offset[d]) / moving->spacing[d]);
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
	    blm->landvox_fix[i*3 + d] = ROUND_INT (
		(blm->fixed_landmarks->points[i*3 + d] 
		    - fixed->offset[d])	/ fixed->spacing[d]);
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
    printf ("Adjusting complete.\n");
}

Bspline_landmarks*
bspline_landmarks_load (char *fixed_fn, char *moving_fn)
{
    int num_fixed_landmarks, num_moving_landmarks;
    Raw_pointset *fixed_landmarks, *moving_landmarks;
    Bspline_landmarks *blm;

    fixed_landmarks = pointset_load (fixed_fn);
    moving_landmarks = pointset_load (moving_fn);

	num_fixed_landmarks = fixed_landmarks->num_points;
	num_moving_landmarks = moving_landmarks->num_points;

#if defined (commentout)
    bspline_landmarks_load_file (&fixed_landmarks, &num_fixed_landmarks, 
	fixed_fn);
    bspline_landmarks_load_file (&moving_landmarks, &num_moving_landmarks, 
	moving_fn);
#endif

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

void bspline_landmarks_write_file (
    const char *fn, char *title, float *coords, int n)
{
    FILE *fp;
    int lidx;
    fp = fopen(fn,"w");
    if (!fp) {
	print_and_exit ("cannot write landmarks to file: %s\n", fn);
    }
    fprintf(fp,"# name = %s\n",title);

    /* Changing LPS to RAS */
    for(lidx=0;lidx<n;lidx++)
	fprintf(fp, "W%d,%f,%f,%f,1,1\n", lidx,
	    -coords[0+3*lidx], 
	    -coords[1+3*lidx],  
	    +coords[2+3*lidx] );

    fclose(fp);
}

/*
Landmark matching implementation "b"
M = moving landmark, F = fixed landmark
Solve  x+u(x)=M for x, then diff = (F - (M-u(x)))
Mar 30 2010 - NSh 
*/
void
bspline_landmarks_score_b (
    Bspline_parms *parms, 
    Bspline_state *bst, 
    Bspline_xform* bxf, 
    Volume *fixed, 
    Volume *moving
)
{
    Bspline_score* ssd = &bst->ssd;
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

    logfile_printf("landmark stiffness is %f\n", parms->landmark_stiffness);

    dd_min = (float *)malloc( blm->num_landmarks * sizeof(float));
    for(d=0;d<blm->num_landmarks;d++) dd_min[d] = 1e20F; //a very large number

    land_p = (int *)malloc( 3* blm->num_landmarks * sizeof(int));
    land_q = (int *)malloc( 3* blm->num_landmarks * sizeof(int));

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
		qidx = volume_index (bxf->vox_per_rgn, q);
		bspline_interp_pix (dxyz, bxf, p, qidx);

		/* Find correspondence in moving image */
		mx = fx + dxyz[0];
		mi = ROUND_INT ((mx - moving->offset[0]) / moving->spacing[0]);
		if (mi < 0 || mi >= moving->dim[0]) continue;
		my = fy + dxyz[1];
		mj = ROUND_INT ((my - moving->offset[1]) / moving->spacing[1]);
		if (mj < 0 || mj >= moving->dim[1]) continue;
		mz = fz + dxyz[2];
		mk = ROUND_INT ((mz - moving->offset[2]) / moving->spacing[2]);
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
    fp  = fopen("warplist_b.fcsv","w");
    fp2 = fopen("distlist_b.dat","w");
    fprintf(fp,"# name = warped\n");

    for( lidx = 0; lidx < blm->num_landmarks; lidx++) {

	//printf("at landmark %d: dd %.1f  dxyz  %.2f %.2f %.2f\n", lidx, dd_min[lidx], dxyz[0],dxyz[1],dxyz[2] );

	if (dd_min[lidx]>10) logfile_printf("Landmark WARNING: landmark far from nearest voxel\n");

	for (d = 0; d < 3; d++) p[d] = land_p[3*lidx+d];
	for (d = 0; d < 3; d++) q[d] = land_q[3*lidx+d];
	qidx = volume_index (bxf->vox_per_rgn, q);
	bspline_interp_pix (dxyz, bxf, p, qidx);

	//actually move the moving landmark; note the minus sign
	for (d = 0; d < 3; d++) {
	    blm->warped_landmarks[3*lidx+d] 
		= blm->moving_landmarks->points[3*lidx + d] - dxyz[d];
	}
	for (d = 0; d < 3; d++) {
	    diff[d] = blm->fixed_landmarks->points[3*lidx+d]
		-blm->warped_landmarks[3*lidx+d];
	}
	l_dist = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
	land_score += l_dist;
	land_rawdist += sqrt(l_dist);
	for (d = 0; d < 3; d++) dc_dv[d] = land_grad_coeff * diff[d]; 
	bspline_update_grad (bst, bxf, p, qidx, dc_dv);

	// Note: Slicer landmarks are in RAS coordinates. Change LPS to RAS 
	fprintf(fp, "W%d,%f,%f,%f,1,1\n", lidx,
	    -blm->warped_landmarks[0+3*lidx], 
	    -blm->warped_landmarks[1+3*lidx],  
	    +blm->warped_landmarks[2+3*lidx] );
	fprintf(fp2,"W%d %.3f\n", lidx, sqrt(l_dist));
    }

    fclose(fp);
    fclose(fp2);

    free(dd_min);
    free(land_p);
    free(land_q);

    land_score = land_score * parms->landmark_stiffness / blm->num_landmarks;
    printf ("        LM DIST %.4f COST %.4f\n", land_rawdist, land_score);
    ssd->score += land_score;
}
#endif

void
bspline_landmarks_score_a (
    Bspline_parms *parms, 
    Bspline_state *bst, 
    Bspline_xform* bxf, 
    Volume *fixed, 
    Volume *moving
)
{
    Bspline_score* ssd = &bst->ssd;
    Bspline_landmarks *blm = &parms->blm;
    int lidx;
    FILE *fp, *fp2;
    float land_score, land_grad_coeff, land_rawdist;

    land_score = 0;
    land_rawdist = 0;
    //land_grad_coeff = parms->landmark_stiffness / blm->num_landmarks;
    land_grad_coeff = blm->landmark_stiffness / blm->num_landmarks;

    logfile_printf ("landmark stiffness is %f\n", blm->landmark_stiffness);

    fp  = fopen ("warplist_a.fcsv","w");
    fp2 = fopen ("distlist_a.dat","w");
    fprintf (fp,"# name = warped\n");

    for (lidx=0; lidx < blm->num_landmarks; lidx++)
    {
	int d;
	int p[3], q[3];
	int qidx;
	float mxyz[3];   /* Location of fixed landmark in moving image */
	float diff[3];   /* mxyz - moving_landmark */
	float dc_dv[3];
	float dxyz[3];
	float l_dist=0;

	for (d = 0; d < 3; d++) {
	    p[d] = blm->landvox_fix[lidx*3+d] / bxf->vox_per_rgn[d];
	    q[d] = blm->landvox_fix[lidx*3+d] % bxf->vox_per_rgn[d];
	}

        qidx = volume_index (bxf->vox_per_rgn, q);
        bspline_interp_pix (dxyz, bxf, p, qidx);

#if defined (commentout)
	/* FIX */
	for (d = 0; d < 3; d++) {
	    mxyz[d] = blm->fixed_landmarks->points[lidx*3+d] + dxyz[d];
	    diff[d] = blm->moving_landmarks->points[lidx*3+d] - mxyz[d];
	}
#endif

#if defined (commentout)
	printf ("    flm = %f %f %f\n", blm->fixed_landmarks[lidx*3+0], 
	    blm->fixed_landmarks[lidx*3+1], blm->fixed_landmarks[lidx*3+2]);
	printf ("    mxyz = %f %f %f\n", mxyz[0], mxyz[1], mxyz[2]);
	printf ("    mlm = %f %f %f\n", blm->moving_landmarks[lidx*3+0], 
	    blm->moving_landmarks[lidx*3+1], blm->moving_landmarks[lidx*3+2]);
#endif

        l_dist = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];

        land_score += l_dist;
        land_rawdist += sqrt(l_dist);

        // calculating gradients
        dc_dv[0] = - land_grad_coeff * diff[0];
        dc_dv[1] = - land_grad_coeff * diff[1];
        dc_dv[2] = - land_grad_coeff * diff[2];
        bspline_update_grad (bst, bxf, p, qidx, dc_dv);

	/* Note: Slicer landmarks are in RAS coordinates. Change LPS to RAS */
        fprintf (fp, "W%d,%f,%f,%f,1,1\n", lidx, -mxyz[0], -mxyz[1], mxyz[2]);
        fprintf (fp2,"W%d %.3f\n", lidx, sqrt(l_dist));
    }
    fclose(fp);
    fclose(fp2);

    land_score = land_score * blm->landmark_stiffness / blm->num_landmarks;
    printf ("        LM DIST %.4f COST %.4f\n", land_rawdist, land_score);
    ssd->score += land_score;
}

void
bspline_landmarks_score (
    Bspline_parms *parms, 
    Bspline_state *bst, 
    Bspline_xform* bxf, 
    Volume *fixed, 
    Volume *moving
)
{
    /* Only 'a' is supported at this time */
    bspline_landmarks_score_a (parms, bst, bxf, fixed, moving);
}

#if defined (commentout)
/*
Moves moving landmarks according to the current vector field.
Output goes into warped_landmarks and landvox_warp
LW = warped landmark
We must solve LW + u(LW) = LM to get new LW, corresponding to current vector field.
*/
void bspline_landmarks_warp (
    Volume *vector_field, 
    Bspline_parms *parms,
    Bspline_xform* bxf, 
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

    if (vector_field->pix_type != PT_VF_FLOAT_INTERLEAVED)
	print_and_exit ("Sorry, this type of vector field is not supported in landmarks_warp\n");	
    vf = (float *)vector_field->img;

    dd_min = (float *)malloc( blm->num_landmarks * sizeof(float));
    for (d=0;d<blm->num_landmarks;d++) dd_min[d] = 1e20F; //a very large number

    for (rk = 0, fk = bxf->roi_offset[2]; rk < bxf->roi_dim[2]; rk++, fk++) {
	fz = bxf->img_origin[2] + bxf->img_spacing[2] * fk;
	for (rj = 0, fj = bxf->roi_offset[1]; rj < bxf->roi_dim[1]; rj++, fj++) {
	    fy = bxf->img_origin[1] + bxf->img_spacing[1] * fj;
	    for (ri = 0, fi = bxf->roi_offset[0]; ri < bxf->roi_dim[0]; ri++, fi++) {
		fx = bxf->img_origin[0] + bxf->img_spacing[0] * fi;

		fv = fk * vector_field->dim[0] * vector_field->dim[1] 
		    + fj * vector_field->dim[0] +fi ;

		for (d=0;d<3;d++) dxyz[d] = vf[3*fv+d];

		/* Find correspondence in moving image */
		mx = fx + dxyz[0];
		mi = ROUND_INT ((mx - moving->offset[0]) / moving->spacing[0]);
		if (mi < 0 || mi >= moving->dim[0]) continue;
		my = fy + dxyz[1];
		mj = ROUND_INT ((my - moving->offset[1]) / moving->spacing[1]);
		if (mj < 0 || mj >= moving->dim[1]) continue;
		mz = fz + dxyz[2];
		mk = ROUND_INT ((mz - moving->offset[2]) / moving->spacing[2]);
		if (mk < 0 || mk >= moving->dim[2]) continue;

		//saving vector field in a voxel which is the closest to landvox_mov
		//after being displaced by the vector field
		for (lidx = 0; lidx < blm->num_landmarks; lidx++) {
		    dd = (mi - blm->landvox_mov[lidx*3+0]) * (mi - blm->landvox_mov[lidx*3+0])
			+(mj - blm->landvox_mov[lidx*3+1]) * (mj - blm->landvox_mov[lidx*3+1])
			+(mk - blm->landvox_mov[lidx*3+2]) * (mk - blm->landvox_mov[lidx*3+2]);
		    if (dd < dd_min[lidx]) { 
			dd_min[lidx]=dd;   
			for (d=0;d<3;d++) {
			    blm->landmark_dxyz[3*lidx+d] = vf[3*fv+d];
			}
		    }
		} 
	    }
	}
    }

    for (i=0;i<blm->num_landmarks;i++)  {
	for (d=0; d<3; d++) {
	    blm->warped_landmarks[3*i+d] 
		= blm->moving_landmarks->points[3*i+d]
		- blm->landmark_dxyz[3*i+d];
	}
    }

    /* calculate voxel positions of warped landmarks  */
    for (lidx = 0; lidx < blm->num_landmarks; lidx++) {
	for (d = 0; d < 3; d++) {
	    blm->landvox_warp[lidx*3 + d] 
		= ROUND_INT ((blm->warped_landmarks[lidx*3 + d] 
			- fixed->offset[d]) / fixed->spacing[d]);
pp	    if (blm->landvox_warp[lidx*3 + d] < 0 
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
    free (dd_min);
}
#endif
