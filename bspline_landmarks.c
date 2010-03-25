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
    if (blm->landvox_fix) {
	free (blm->landvox_fix);
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
bspline_landmarks_adjust (Bspline_landmarks *blm, Volume *fixed)
{
    int i, d;

    // Save position of landmarks in voxels for vector field calculation
    // GCS: I think we only need fixed
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

    printf ("Found %d landmark(s) from fixed to moving image\n", 
	blm->num_landmarks);

    return blm;
}

/* 
NSh Mean-square error registration with landmarks.
Based on bspline_score_a_mse

Input files: 
-fixed.fcsv, moving.fcsv: fiducials files from Slicer3
-stiffness.txt: one line, stiffness of landmark-to-landmark spring

Output files: warplist.fcsv is a fiducials file with landmarks 
on the warped image; distlist.dat are the distances between
corresponding landmarks in fixed and warped images.

Parameter: land_coeff = spring constant for attraction
between landmarks. land_coeff = 0 corresponds to no constraints
on landmarks, exactly as in bspline_score_a_mse.

Mar 15 2010 - NSh 
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

    land_score = 0;
    land_rawdist = 0;
    land_grad_coeff = parms->landmark_stiffness / blm->num_landmarks;

    fp  = fopen("warplist.fcsv","w");
    fp2 = fopen("distlist.dat","w");
    fprintf(fp,"# name = warped\n");

    for (lidx=0; lidx < blm->num_landmarks; lidx++)
    {
	int d;
	int p[3], q[3];
	int qidx;
	float mxyz[3];   /* Location of fixed landmark in moving image */
	float diff[3];   /* mxyz - moving_landmark */
	float dc_dv[3];
	float dxyz[3];
	float l_dist;

	for (d = 0; d < 3; d++) {
	    p[d] = blm->landvox_fix[lidx*3+d] / bxf->vox_per_rgn[d];
	    q[d] = blm->landvox_fix[lidx*3+d] % bxf->vox_per_rgn[d];
	}

        qidx = INDEX_OF (q, bxf->vox_per_rgn);
        bspline_interp_pix (dxyz, bxf, p, qidx);

	for (d = 0; d < 3; d++) {
	    mxyz[d] = blm->fixed_landmarks[lidx*3+d] + dxyz[d];
	    diff[d] = blm->moving_landmarks[lidx*3+d] - mxyz[d];
	}

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
        dc_dv[0] = land_grad_coeff * diff[0];
        dc_dv[1] = land_grad_coeff * diff[1];
        dc_dv[2] = land_grad_coeff * diff[2];
        bspline_update_grad (bst, bxf, p, qidx, dc_dv);

	/* Note: Slicer landmarks are in RAS coordinates. Change LPS to RAS */
        fprintf (fp, "W%d,%f,%f,%f,1,1\n", lidx, -mxyz[0], -mxyz[1], mxyz[2]);
        fprintf (fp2,"W%d %.3f\n", lidx, sqrt(l_dist));
    }
    fclose(fp);
    fclose(fp2);

    land_score = land_score * parms->landmark_stiffness / blm->num_landmarks;
    printf ("        LM DIST %.4f COST %.4f\n", land_rawdist, land_score);
    ssd->score += land_score;
}
