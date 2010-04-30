/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include "math_util.h"
#include "volume.h"
#include "volume_limit.h"

#define DRR_BOUNDARY_TOLERANCE 1e-6
#define DRR_LEN_TOLERANCE 1e-6

static Point_location
test_boundary (Volume_limit* vol_limit, int d, double x)
{
    if (vol_limit->dir[d] == 0) {
	if (x < vol_limit->limits[d][0]) {
	    return POINTLOC_LEFT;
	} else if (x > vol_limit->limits[d][1]) {
	    return POINTLOC_RIGHT;
	} else {
	    return POINTLOC_INSIDE;
	}
    } else {
	if (x > vol_limit->limits[d][0]) {
	    return POINTLOC_LEFT;
	} else if (x < vol_limit->limits[d][1]) {
	    return POINTLOC_RIGHT;
	} else {
	    return POINTLOC_INSIDE;
	}
    }
}

/* Return 1 if line segment intersects boundary */
int
volume_limit_clip_segment (
    Volume_limit *vol_limit,    /* INPUT:  The bounding box to clip to */
    double *ip1,                /* OUTPUT: Intersection point 1 */
    double *ip2,                /* OUTPUT: Intersection point 2 */
    double *p1,                 /* INPUT:  Line segment point 1 */
    double *p2                  /* INPUT:  Line segment point 2 */
)
{
    Point_location ploc[3][2];
    double ray[3];
    double alpha[3][2];
    double alpha_in, alpha_out;
    int d;

    for (d = 0; d < 3; d++) {
	ploc[d][0] = test_boundary (vol_limit, d, p1[d]);
	ploc[d][1] = test_boundary (vol_limit, d, p2[d]);
	/* Immediately reject segments which don't intersect the volume in 
	   this dimension */
	if (ploc[d][0] == POINTLOC_LEFT && ploc[d][1] == POINTLOC_LEFT) {
	    return 0;
	}
	if (ploc[d][0] == POINTLOC_RIGHT && ploc[d][1] == POINTLOC_RIGHT) {
	    return 0;
	}
    }

#if defined (ULTRA_VERBOSE)
    printf ("vol_limit[*][0] = %g %g %g\n", vol_limits[0].limits[0], 
	vol_limits[1].limits[0], vol_limits[2].limits[0]);
    printf ("vol_limit[*][1] = %g %g %g\n", vol_limits[0].limits[1], 
	vol_limits[1].limits[1], vol_limits[2].limits[1]);
    printf ("ploc[*][0]: %d %d %d\n", ploc[0][0], ploc[1][0], ploc[2][0]);
    printf ("ploc[*][1]: %d %d %d\n", ploc[0][1], ploc[1][1], ploc[2][1]);
#endif

    /* If we made it here, all three dimensions have some range of alpha
       where they intersects the volume.  However, these alphas might 
       not overlap.  We compute the alphas, then test overlapping 
       alphas to find the segment range within the volume.  */
    for (d = 0; d < 3; d++) {
	if (ploc[d][0] == POINTLOC_LEFT) {
	    alpha[d][0] = (vol_limit->limits[d][0] - p1[d]) / (p2[d] - p1[d]);
	} else if (ploc[d][0] == POINTLOC_RIGHT) {
	    alpha[d][0] = (p1[d] - vol_limit->limits[d][1]) / (p1[d] - p2[d]);
	} else {
	    alpha[d][0] = 0.0;
	}
	if (ploc[d][1] == POINTLOC_LEFT) {
	    alpha[d][1] = (vol_limit->limits[d][0] - p1[d]) / (p2[d] - p1[d]);
	} else if (ploc[d][1] == POINTLOC_RIGHT) {
	    alpha[d][1] = (p1[d] - vol_limit->limits[d][1]) / (p1[d] - p2[d]);
	} else {
	    alpha[d][1] = 1.0;
	}
    }

    /* alpha_in is the alpha where the segment enters the boundary, and 
       alpha_out is where it exits the boundary.  */
    alpha_in = alpha[0][0];
    alpha_out = alpha[0][1];
    for (d = 1; d < 3; d++) {
	if (alpha_in < alpha[d][0]) alpha_in = alpha[d][0];
	if (alpha_out > alpha[d][1]) alpha_out = alpha[d][1];
    }
#if defined (ULTRA_VERBOSE)
    printf ("alpha[*][0] = %g %g %g\n", alpha[0][0], alpha[1][0], alpha[2][0]);
    printf ("alpha[*][1] = %g %g %g\n", alpha[0][1], alpha[1][1], alpha[2][1]);
    printf ("alpha in/out = %g %g\n", alpha_in, alpha_out);
#endif

    /* If exit is before entrance, the segment does not intersect the volume */
    if (alpha_out - alpha_in < DRR_LEN_TOLERANCE) {
	return 0;
    }

    /* Create the volume intersection points */
    vec3_sub3 (ray, p2, p1);
    for (d = 0; d < 3; d++) {
	ip1[d] = p1[d] + alpha_in * ray[d];
	ip2[d] = p1[d] + alpha_out * ray[d];
    }

    return 1;
}

void
volume_limit_set (Volume_limit *vol_limit, Volume *vol)
{
    int d;

    /* Compute volume boundary box */
    for (d = 0; d < 3; d++) {
	vol_limit->limits[d][0] = vol->offset[d] - 0.5 * vol->pix_spacing[d];
	vol_limit->limits[d][1] = vol_limit->limits[d][0] 
	    + vol->dim[d] * vol->pix_spacing[d];
	if (vol_limit->limits[d][0] <= vol_limit->limits[d][1]) {
	    vol_limit->dir[d] = 0;
	    vol_limit->limits[d][0] += DRR_BOUNDARY_TOLERANCE;
	    vol_limit->limits[d][1] -= DRR_BOUNDARY_TOLERANCE;
	} else {
	    vol_limit->dir[d] = 1;
	    vol_limit->limits[d][0] -= DRR_BOUNDARY_TOLERANCE;
	    vol_limit->limits[d][1] += DRR_BOUNDARY_TOLERANCE;
	}
    }
}
