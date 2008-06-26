/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -------------------------------------------------------------------------
    REFS:
    http://en.wikipedia.org/wiki/B-spline
    http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bspline-construct.html
    http://graphics.idav.ucdavis.edu/education/CAGDNotes/Quadratic-B-Spline-Surface-Refinement/Quadratic-B-Spline-Surface-Refinement.html

    ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "volume.h"
#include "readmha.h"
#include "bspline_opts.h"
#include "bspline.h"

#define round_int(x) ((x)>=0?(long)((x)+0.5):(long)(-(-(x)+0.5)))

void
control_point_loop (BSPLINE_Data* bspd, Volume* fixed, BSPLINE_Parms* parms)
{
    int i, j, k;
    int rx, ry, rz;
    int vx, vy, vz;
    int cidx;
    float* img;

    img = (float*) fixed->img;

    /* Loop through cdim^3 control points */
    for (k = 0; k < bspd->cdims[2]; k++) {
	for (j = 0; j < bspd->cdims[1]; j++) {
	    for (i = 0; i < bspd->cdims[0]; i++) {

		/* Linear index of control point */
		cidx = k * bspd->cdims[1] * bspd->cdims[0]
		    + j * bspd->cdims[0] + i;

		/* Each control point has 64 regions */
		for (rz = 0; rz < 4; rz ++) {
		    for (ry = 0; ry < 4; ry ++) {
			for (rx = 0; rx < 4; rx ++) {

			    /* Some of the 64 regions are invalid. */
			    if (k + rz - 2 < 0) continue;
			    if (k + rz - 2 >= bspd->rdims[2]) continue;
			    if (j + ry - 2 < 0) continue;
			    if (j + ry - 2 >= bspd->rdims[1]) continue;
			    if (i + rx - 2 < 0) continue;
			    if (i + rx - 2 >= bspd->rdims[0]) continue;

			    /* Each region has int_spacing^3 voxels */
			    for (vz = 0; vz < parms->int_spacing[2]; vz ++) {
				for (vy = 0; vy < parms->int_spacing[1]; vy ++) {
				    for (vx = 0; vx < parms->int_spacing[0]; vx ++) {
					int img_idx[3], p;
					float img_val, coeff_val;

					/* Get (i,j,k) index of the voxel */
					img_idx[0] = (i + rx - 2) * parms->int_spacing[0] + vx;
					img_idx[1] = (j + ry - 2) * parms->int_spacing[1] + vy;
					img_idx[2] = (k + rz - 2) * parms->int_spacing[2] + vz;

					/* Some of the pixels are invalid. */
					if (img_idx[0] > fixed->dim[0]) continue;
					if (img_idx[1] > fixed->dim[1]) continue;
					if (img_idx[2] > fixed->dim[2]) continue;

					/* Get the image value */
					p = img_idx[2] * fixed->dim[1] * fixed->dim[0] 
					    + img_idx[1] * fixed->dim[0] + img_idx[0];
					img_val = img[p];

					/* Get coefficient multiplier */
					p = vz * parms->int_spacing[0] * parms->int_spacing[1]
					    + vy * parms->int_spacing[0] + vx;
					coeff_val = bspd->coeff[p];

					/* Here you would update the gradient: 
					    grad[cidx] += (fixed_val - moving_val) * coeff_val;
					*/
				    }
				}
			    }
			}
		    }
		}
	    }
	}
    }
}


