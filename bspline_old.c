#include "bspline_old.h"

//////////////////////////////////////////////////////////////////////
// This file contains depricated functions relevant to B-spline
//////////////////////////////////////////////////////////////////////

static void
bspline_interp_pix_b (float out[3], BSPLINE_Xform* bxf, int pidx, int qidx)
{
    int i, j, k, m;
    int cidx;
    float* q_lut = &bxf->q_lut[qidx*64];
    int* c_lut = &bxf->c_lut[pidx*64];

    out[0] = out[1] = out[2] = 0;
    m = 0;
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		cidx = 3 * c_lut[m];
		out[0] += q_lut[m] * bxf->coeff[cidx+0];
		out[1] += q_lut[m] * bxf->coeff[cidx+1];
		out[2] += q_lut[m] * bxf->coeff[cidx+2];
		m ++;
	    }
	}
    }
}



/* Clipping is done using clamping.  You should have "air" as the outside
   voxel so pixels can be clamped to air.  */
inline void
clip_and_interpolate_obsolete (
    Volume* moving,	/* Moving image */
    float* dxyz,	/* Vector displacement of current voxel */
    float* dxyzf,	/* Floor of vector displacement */
    int d,		/* 0 for x, 1 for y, 2 for z */
    int* maf,		/* x, y, or z coord of "floor" pixel in moving img */
    int* mar,		/* x, y, or z coord of "round" pixel in moving img */
    int a,		/* Index of base voxel (before adding displacemnt) */
    float* fa1,		/* Fraction of interpolant for lower index voxel */
    float* fa2		/* Fraction of interpolant for upper index voxel */
)
{
    dxyzf[d] = floor (dxyz[d]);
    *maf = a + (int) dxyzf[d];
    *mar = a + ROUND_INT (dxyz[d]);
    *fa2 = dxyz[d] - dxyzf[d];
    if (*maf < 0) {
	*maf = 0;
	*mar = 0;
	*fa2 = 0.0f;
    } else if (*maf >= moving->dim[d] - 1) {
	*maf = moving->dim[d] - 2;
	*mar = moving->dim[d] - 1;
	*fa2 = 1.0f;
    }
    *fa1 = 1.0f - *fa2;
}


/* bspline_score_f_mse:
 * This version is similar to version "D" in that it calculates the
 * score tile by tile. For the gradient, it iterates over each
 * control knot, determines which tiles influence the knot, and then
 * sums the total influence from each of those tiles. This implementation
 * allows for greater parallelization on the GPU.
 */
void bspline_score_f_mse (BSPLINE_Parms *parms,
			  Bspline_state *bst, 
			  BSPLINE_Xform *bxf,
			  Volume *fixed,
			  Volume *moving,
			  Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;

    int i;
    int qz;
    int p[3];
    float* dc_dv;
    float* f_img = (float*)fixed->img;
    float* m_img = (float*)moving->img;
    float* m_grad = (float*)moving_grad->img;
    int num_vox;
    int cidx;
    Timer timer;
    double interval;

    int total_vox_per_rgn = bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2];

    //static int it = 0;
    //char debug_fn[1024];

    plm_timer_start (&timer);

    // Allocate memory for the dc_dv array. In this implementation, 
    // dc_dv values are computed for each voxel, and stored in the 
    // first phase.  Then the second phase uses the dc_dv calculation.
    dc_dv = (float*) malloc (3 * total_vox_per_rgn 
			     * bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2] 
			     * sizeof(float));

    ssd->score = 0;
    memset(ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;

    // Serial across tiles
    for (p[2] = 0; p[2] < bxf->rdims[2]; p[2]++) {
	for (p[1] = 0; p[1] < bxf->rdims[1]; p[1]++) {
	    for (p[0] = 0; p[0] < bxf->rdims[0]; p[0]++) {
		int tile_num_vox = 0;
		double tile_score = 0.0;
		int pidx;
		int* c_lut;

		// Compute linear index for tile
		pidx = INDEX_OF (p, bxf->rdims);

		// Find c_lut row for this tile
		c_lut = &bxf->c_lut[pidx*64];

		// Parallel across offsets within the tile
#pragma omp parallel for reduction (+:tile_num_vox,tile_score)
		for (qz = 0; qz < bxf->vox_per_rgn[2]; qz++) {
		    int q[3];
		    q[2] = qz;
		    for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
			for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {
			    float diff;
			    int qidx;
			    int fi, fj, fk, fv;
			    float mi, mj, mk;
			    float fx, fy, fz;
			    float mx, my, mz;
			    int mif, mjf, mkf, mvf;  /* Floor */
			    int mir, mjr, mkr, mvr;  /* Round */
			    float* dc_dv_row;
			    float dxyz[3];
			    float fx1, fx2, fy1, fy2, fz1, fz2;
			    float m_val;
			    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
			    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

			    // Compute linear index for this offset
			    qidx = INDEX_OF (q, bxf->vox_per_rgn);

			    // Tentatively mark this pixel as no contribution
			    dc_dv[(pidx * 3 * total_vox_per_rgn) + (3*qidx+0)] = 0.f;
			    dc_dv[(pidx * 3 * total_vox_per_rgn) + (3*qidx+1)] = 0.f;
			    dc_dv[(pidx * 3 * total_vox_per_rgn) + (3*qidx+2)] = 0.f;

			    // Get (i,j,k) index of the voxel
			    fi = bxf->roi_offset[0] + p[0] * bxf->vox_per_rgn[0] + q[0];
			    fj = bxf->roi_offset[1] + p[1] * bxf->vox_per_rgn[1] + q[1];
			    fk = bxf->roi_offset[2] + p[2] * bxf->vox_per_rgn[2] + q[2];

			    // Some of the pixels are outside image
			    if (fi >= bxf->roi_offset[0] + bxf->roi_dim[0]) continue;
			    if (fj >= bxf->roi_offset[1] + bxf->roi_dim[1]) continue;
			    if (fk >= bxf->roi_offset[2] + bxf->roi_dim[2]) continue;

			    //if (p[2] == 4) {
			    //	logfile_printf ("Kernel 1, pix %d %d %d\n", fi, fj, fk);
			    //	logfile_printf ("Kernel 1, offset %d %d %d\n", q[0], q[1], q[2]);
			    //}

			    // Compute physical coordinates of fixed image voxel
			    fx = bxf->img_origin[0] + bxf->img_spacing[0] * fi;
			    fy = bxf->img_origin[1] + bxf->img_spacing[1] * fj;
			    fz = bxf->img_origin[2] + bxf->img_spacing[2] * fk;

			    // Compute linear index of fixed image voxel
			    fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

			    // Get B-spline deformation vector
			    bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

			    // Find correspondence in moving image
			    mx = fx + dxyz[0];
			    mi = (mx - moving->offset[0]) / moving->pix_spacing[0];
			    if (mi < -0.5 || mi > moving->dim[0] - 0.5) continue;

			    my = fy + dxyz[1];
			    mj = (my - moving->offset[1]) / moving->pix_spacing[1];
			    if (mj < -0.5 || mj > moving->dim[1] - 0.5) continue;

			    mz = fz + dxyz[2];
			    mk = (mz - moving->offset[2]) / moving->pix_spacing[2];
			    if (mk < -0.5 || mk > moving->dim[2] - 0.5) continue;

			    // Compute interpolation fractions
			    clamp_linear_interpolate_inline (mi, moving->dim[0]-1, &mif, &mir, &fx1, &fx2);
			    clamp_linear_interpolate_inline (mj, moving->dim[1]-1, &mjf, &mjr, &fy1, &fy2);
			    clamp_linear_interpolate_inline (mk, moving->dim[2]-1, &mkf, &mkr, &fz1, &fz2);

			    // Compute moving image intensity using linear interpolation
			    mvf = (mkf * moving->dim[1] + mjf) * moving->dim[0] + mif;
			    m_x1y1z1 = fx1 * fy1 * fz1 * m_img[mvf];
			    m_x2y1z1 = fx2 * fy1 * fz1 * m_img[mvf+1];
			    m_x1y2z1 = fx1 * fy2 * fz1 * m_img[mvf+moving->dim[0]];
			    m_x2y2z1 = fx2 * fy2 * fz1 * m_img[mvf+moving->dim[0]+1];
			    m_x1y1z2 = fx1 * fy1 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]];
			    m_x2y1z2 = fx2 * fy1 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+1];
			    m_x1y2z2 = fx1 * fy2 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]];
			    m_x2y2z2 = fx2 * fy2 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]+1];
			    m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 
				    + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

			    // Compute intensity difference
			    diff = f_img[fv] - m_val;

			    // We'll go ahead and accumulate the score here, but you would 
			    // have to reduce somewhere else instead.
			    tile_score += diff * diff;
			    tile_num_vox++;

			    // Compute spatial gradient using nearest neighbors
			    mvr = (mkr * moving->dim[1] + mjr) * moving->dim[0] + mir;

			    // Store dc_dv for this offset
			    dc_dv_row = &dc_dv[3 * total_vox_per_rgn * pidx];
			    dc_dv_row[3*qidx+0] = diff * m_grad[3*mvr+0];  // x component
			    dc_dv_row[3*qidx+1] = diff * m_grad[3*mvr+1];  // y component
			    dc_dv_row[3*qidx+2] = diff * m_grad[3*mvr+2];  // z component
			}
		    }
		}
		ssd->score += tile_score;
		num_vox += tile_num_vox;
	    }
	}
    }

    /* Parallel across control knots */
#pragma omp parallel for
    for (cidx = 0; cidx < bxf->num_knots; cidx++) {
	int knot_x, knot_y, knot_z;
	int x_offset, y_offset, z_offset;

	// Determine the x, y, and z offset of the knot within the grid.
	knot_x = cidx % bxf->cdims[0];
	knot_y = ((cidx - knot_x) / bxf->cdims[0]) % bxf->cdims[1];
	knot_z = ((((cidx - knot_x) / bxf->cdims[0]) - knot_y) / bxf->cdims[1]) % bxf->cdims[2];

	// Subtract 1 from each of the knot indices to account for the differing origin
	// between the knot grid and the tile grid.
	knot_x -= 1;
	knot_y -= 1;
	knot_z -= 1;

	// Iterate through each of the 64 tiles that influence this control knot.
	for (z_offset = -2; z_offset < 2; z_offset++) {
	    for (y_offset = -2; y_offset < 2; y_offset++) {
		for (x_offset = -2; x_offset < 2; x_offset++) {

		    // Using the current x, y, and z offset from the control knot position,
		    // calculate the index for one of the tiles that influence this knot.
		    int tile_x, tile_y, tile_z;

		    tile_x = knot_x + x_offset;
		    tile_y = knot_y + y_offset;
		    tile_z = knot_z + z_offset;

		    // Determine if the tile lies within the volume.
		    if((tile_x >= 0 && tile_x < bxf->rdims[0]) &&
		       (tile_y >= 0 && tile_y < bxf->rdims[1]) &&
		       (tile_z >= 0 && tile_z < bxf->rdims[2])) {

			int m;
			int q[3];
			int qidx;
			int pidx;
			int* c_lut;
			float* dc_dv_row;

			// Compute linear index for tile.
			pidx = ((tile_z * bxf->rdims[1] + tile_y) * bxf->rdims[0]) + tile_x;

			// Find c_lut row for this tile.
			c_lut = &bxf->c_lut[64*pidx];

			// Pull out the dc_dv values for just this tile.
			dc_dv_row = &dc_dv[3 * total_vox_per_rgn * pidx];

			// Find the coefficient index in the c_lut row in order to determine
			// the linear index of the control point with respect to the current tile.
			for (m = 0; m < 64; m++) {
			    if (c_lut[m] == cidx) {
				break;
			    }
			}

			// Iterate through each of the offsets within the tile and 
			// accumulate the gradient.
			for (qidx = 0, q[2] = 0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
			    for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
				for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++, qidx++) {

				    // Find q_lut row for this offset.
				    float* q_lut = &bxf->q_lut[64*qidx];							

				    // Accumulate update to gradient for this control point.
				    ssd->grad[3*cidx+0] += dc_dv_row[3*qidx+0] * q_lut[m];
				    ssd->grad[3*cidx+1] += dc_dv_row[3*qidx+1] * q_lut[m];
				    ssd->grad[3*cidx+2] += dc_dv_row[3*qidx+2] * q_lut[m];
				}
			    }
			}
		    }
		}
	    }
	}
    }

    free (dc_dv);

    /* Normalize score for MSE */
    ssd->score = ssd->score / num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}

void
bspline_score_e_mse (BSPLINE_Parms *parms, 
		     Bspline_state *bst,
		     BSPLINE_Xform* bxf, 
		     Volume *fixed, 
		     Volume *moving, 
		     Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i;
    int p[3];
    int s[3];
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    int num_vox;
    Timer timer;
    double interval;

    static int it = 0;
    char debug_fn[1024];
    FILE* fp;

    if (parms->debug) {
	sprintf (debug_fn, "dc_dv_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    plm_timer_start (&timer);

    ssd->score = 0;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;

    /* Serial across 64 groups of tiles */
    for (s[2] = 0; s[2] < 4; s[2]++) {
	for (s[1] = 0; s[1] < 4; s[1]++) {
	    for (s[0] = 0; s[0] < 4; s[0]++) {
		int tile;
		int tilelist_len = 0;
		int *tilelist = 0;
		int tile_num_vox = 0;
		double tile_score = 0.0;

		/* Create list of tiles in the group */
		for (p[2] = s[2]; p[2] < bxf->rdims[2]; p[2]+=4) {
		    for (p[1] = s[1]; p[1] < bxf->rdims[1]; p[1]+=4) {
			for (p[0] = s[0]; p[0] < bxf->rdims[0]; p[0]+=4) {
			    tilelist_len ++;
			    tilelist = realloc (tilelist, 3 * tilelist_len * sizeof(int));
			    tilelist[3*tilelist_len - 3] = p[0];
			    tilelist[3*tilelist_len - 2] = p[1];
			    tilelist[3*tilelist_len - 1] = p[2];
			}
		    }
		}

		//		printf ("%d tiles in parallel...\n");

		/* Parallel across tiles within groups */
#pragma omp parallel for reduction (+:tile_num_vox,tile_score)
		for (tile = 0; tile < tilelist_len; tile++) {
		    int pidx;
		    int* c_lut;
		    int p[3];
		    int q[3];

		    p[0] = tilelist[3*tile + 0];
		    p[1] = tilelist[3*tile + 1];
		    p[2] = tilelist[3*tile + 2];

		    /* Compute linear index for tile */
		    pidx = INDEX_OF (p, bxf->rdims);

		    /* Find c_lut row for this tile */
		    c_lut = &bxf->c_lut[pidx*64];

		    //logfile_printf ("Kernel 1, tile %d %d %d\n", p[0], p[1], p[2]);

		    /* Serial across offsets */
		    for (q[2] = 0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
			for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
			    for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {
				int qidx;
				int fi, fj, fk, fv;
				float mx, my, mz;
				float mi, mj, mk;
				float fx, fy, fz;
				int mif, mjf, mkf, mvf;  /* Floor */
				int mir, mjr, mkr, mvr;  /* Round */
				float fx1, fx2, fy1, fy2, fz1, fz2;
				float dxyz[3];
				float m_val;
				float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
				float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
				float diff;
				float dc_dv[3];
				int m;

				/* Compute linear index for this offset */
				qidx = INDEX_OF (q, bxf->vox_per_rgn);

				/* Get (i,j,k) index of the voxel */
				fi = bxf->roi_offset[0] + p[0] * bxf->vox_per_rgn[0] + q[0];
				fj = bxf->roi_offset[1] + p[1] * bxf->vox_per_rgn[1] + q[1];
				fk = bxf->roi_offset[2] + p[2] * bxf->vox_per_rgn[2] + q[2];

				/* Some of the pixels are outside image */
				if (fi >= bxf->roi_offset[0] + bxf->roi_dim[0]) continue;
				if (fj >= bxf->roi_offset[1] + bxf->roi_dim[1]) continue;
				if (fk >= bxf->roi_offset[2] + bxf->roi_dim[2]) continue;

				/* Compute physical coordinates of fixed image voxel */
				fx = bxf->img_origin[0] + bxf->img_spacing[0] * fi;
				fy = bxf->img_origin[1] + bxf->img_spacing[1] * fj;
				fz = bxf->img_origin[2] + bxf->img_spacing[2] * fk;

				/* Compute linear index of fixed image voxel */
				fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

				/* Get B-spline deformation vector */
				bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

				/* Find correspondence in moving image */
				mx = fx + dxyz[0];
				mi = (mx - moving->offset[0]) / moving->pix_spacing[0];
				if (mi < -0.5 || mi > moving->dim[0] - 0.5) continue;

				my = fy + dxyz[1];
				mj = (my - moving->offset[1]) / moving->pix_spacing[1];
				if (mj < -0.5 || mj > moving->dim[1] - 0.5) continue;

				mz = fz + dxyz[2];
				mk = (mz - moving->offset[2]) / moving->pix_spacing[2];
				if (mk < -0.5 || mk > moving->dim[2] - 0.5) continue;

				/* Compute interpolation fractions */
				clamp_linear_interpolate_inline (mi, moving->dim[0]-1, &mif, &mir, &fx1, &fx2);
				clamp_linear_interpolate_inline (mj, moving->dim[1]-1, &mjf, &mjr, &fy1, &fy2);
				clamp_linear_interpolate_inline (mk, moving->dim[2]-1, &mkf, &mkr, &fz1, &fz2);

				/* Compute moving image intensity using linear interpolation */
				mvf = (mkf * moving->dim[1] + mjf) * moving->dim[0] + mif;
				m_x1y1z1 = fx1 * fy1 * fz1 * m_img[mvf];
				m_x2y1z1 = fx2 * fy1 * fz1 * m_img[mvf+1];
				m_x1y2z1 = fx1 * fy2 * fz1 * m_img[mvf+moving->dim[0]];
				m_x2y2z1 = fx2 * fy2 * fz1 * m_img[mvf+moving->dim[0]+1];
				m_x1y1z2 = fx1 * fy1 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]];
				m_x2y1z2 = fx2 * fy1 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+1];
				m_x1y2z2 = fx1 * fy2 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]];
				m_x2y2z2 = fx2 * fy2 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]+1];
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 
					+ m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				/* Compute intensity difference */
				diff = f_img[fv] - m_val;

				/* We'll go ahead and accumulate the score here, but you would 
				   have to reduce somewhere else instead */
				tile_score += diff * diff;
				tile_num_vox ++;

				/* Compute spatial gradient using nearest neighbors */
				mvr = (mkr * moving->dim[1] + mjr) * moving->dim[0] + mir;

				/* Store dc_dv for this offset */
				dc_dv[0] = diff * m_grad[3*mvr+0];  /* x component */
				dc_dv[1] = diff * m_grad[3*mvr+1];  /* y component */
				dc_dv[2] = diff * m_grad[3*mvr+2];  /* z component */

				/* Serial across 64 control points */
				for (m = 0; m < 64; m++) {
				    float* q_lut;
				    int cidx;

				    /* Find index of control point within coefficient array */
				    cidx = c_lut[m] * 3;

				    /* Find q_lut row for this offset */
				    q_lut = &bxf->q_lut[qidx*64];

				    /* Accumulate update to gradient for this 
				       control point */
				    ssd->grad[cidx+0] += dc_dv[0] * q_lut[m];
				    ssd->grad[cidx+1] += dc_dv[1] * q_lut[m];
				    ssd->grad[cidx+2] += dc_dv[2] * q_lut[m];
				}
			    }
			}
		    }
		}

		num_vox += tile_num_vox;
		ssd->score += tile_score;

		free (tilelist);
	    }
	}
    }

    if (parms->debug) {
	fclose (fp);
    }

    /* Normalize score for MSE */
    ssd->score = ssd->score / num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}


/* Mean-squared error version of implementation "D" */
/* This design could be useful for a low-memory GPU implementation

for each tile in serial

  for each offset in parallel
    for each control point in serial
      accumulate into dxyz
    compute diff
    store dc_dv for this offset

  for each control point in parallel
    for each offset in serial
      accumulate into coefficient

end

The dual design which I didn't use:

for each offset in serial

  for each tile in parallel
    for each control point in serial
      accumulate into dxyz
    compute diff
    store dc_dv for this tile

  for each control point in parallel
    for each offset in serial
      accumulate into coefficient

end
*/

void
bspline_score_d_mse (BSPLINE_Parms *parms, 
		     Bspline_state *bst,
		     BSPLINE_Xform* bxf, 
		     Volume *fixed, 
		     Volume *moving, 
		     Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i;
    int qz;
    int p[3];
    float* dc_dv;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    int num_vox;
    Timer timer;
    double interval;

    static int it = 0;
    char debug_fn[1024];
    FILE* fp;

    if (parms->debug) {
	sprintf (debug_fn, "dc_dv_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    plm_timer_start (&timer);

    dc_dv = (float*) malloc (3*bxf->vox_per_rgn[0]*bxf->vox_per_rgn[1]*bxf->vox_per_rgn[2]*sizeof(float));
    ssd->score = 0;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;

    /* Serial across tiles */
    for (p[2] = 0; p[2] < bxf->rdims[2]; p[2]++) {
	for (p[1] = 0; p[1] < bxf->rdims[1]; p[1]++) {
	    for (p[0] = 0; p[0] < bxf->rdims[0]; p[0]++) {
		int k;
		int pidx;
		int* c_lut;
		int tile_num_vox = 0;
		double tile_score = 0.0;

		/* Compute linear index for tile */
		pidx = INDEX_OF (p, bxf->rdims);

		/* Find c_lut row for this tile */
		c_lut = &bxf->c_lut[pidx*64];

		//logfile_printf ("Kernel 1, tile %d %d %d\n", p[0], p[1], p[2]);

		/* Parallel across offsets */
#pragma omp parallel for reduction (+:tile_num_vox,tile_score)
		for (qz = 0; qz < bxf->vox_per_rgn[2]; qz++) {
		    int q[3];
		    q[2] = qz;
		    for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
			for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {
			    int qidx;
			    int fi, fj, fk, fv;
			    float mx, my, mz;
			    float mi, mj, mk;
			    float fx, fy, fz;
			    int mif, mjf, mkf, mvf;  /* Floor */
			    int mir, mjr, mkr, mvr;  /* Round */
			    float fx1, fx2, fy1, fy2, fz1, fz2;
			    float dxyz[3];
			    float m_val;
			    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
			    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
			    float diff;

			    /* Compute linear index for this offset */
			    qidx = INDEX_OF (q, bxf->vox_per_rgn);

			    /* Tentatively mark this pixel as no contribution */
			    dc_dv[3*qidx+0] = 0.f;
			    dc_dv[3*qidx+1] = 0.f;
			    dc_dv[3*qidx+2] = 0.f;

			    /* Get (i,j,k) index of the voxel */
			    fi = bxf->roi_offset[0] + p[0] * bxf->vox_per_rgn[0] + q[0];
			    fj = bxf->roi_offset[1] + p[1] * bxf->vox_per_rgn[1] + q[1];
			    fk = bxf->roi_offset[2] + p[2] * bxf->vox_per_rgn[2] + q[2];

			    /* Some of the pixels are outside image */
			    if (fi >= bxf->roi_offset[0] + bxf->roi_dim[0]) continue;
			    if (fj >= bxf->roi_offset[1] + bxf->roi_dim[1]) continue;
			    if (fk >= bxf->roi_offset[2] + bxf->roi_dim[2]) continue;

			    //if (p[2] == 4) {
			    //	logfile_printf ("Kernel 1, pix %d %d %d\n", fi, fj, fk);
			    //	logfile_printf ("Kernel 1, offset %d %d %d\n", q[0], q[1], q[2]);
			    //}

			    /* Compute physical coordinates of fixed image voxel */
			    fx = bxf->img_origin[0] + bxf->img_spacing[0] * fi;
			    fy = bxf->img_origin[1] + bxf->img_spacing[1] * fj;
			    fz = bxf->img_origin[2] + bxf->img_spacing[2] * fk;

			    /* Compute linear index of fixed image voxel */
			    fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

			    /* Get B-spline deformation vector */
			    bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

			    /* Find correspondence in moving image */
			    mx = fx + dxyz[0];
			    mi = (mx - moving->offset[0]) / moving->pix_spacing[0];
			    if (mi < -0.5 || mi > moving->dim[0] - 0.5) continue;

			    my = fy + dxyz[1];
			    mj = (my - moving->offset[1]) / moving->pix_spacing[1];
			    if (mj < -0.5 || mj > moving->dim[1] - 0.5) continue;

			    mz = fz + dxyz[2];
			    mk = (mz - moving->offset[2]) / moving->pix_spacing[2];
			    if (mk < -0.5 || mk > moving->dim[2] - 0.5) continue;

			    /* Compute interpolation fractions */
			    clamp_linear_interpolate_inline (mi, moving->dim[0]-1, &mif, &mir, &fx1, &fx2);
			    clamp_linear_interpolate_inline (mj, moving->dim[1]-1, &mjf, &mjr, &fy1, &fy2);
			    clamp_linear_interpolate_inline (mk, moving->dim[2]-1, &mkf, &mkr, &fz1, &fz2);

			    /* Compute moving image intensity using linear interpolation */
			    mvf = (mkf * moving->dim[1] + mjf) * moving->dim[0] + mif;
			    m_x1y1z1 = fx1 * fy1 * fz1 * m_img[mvf];
			    m_x2y1z1 = fx2 * fy1 * fz1 * m_img[mvf+1];
			    m_x1y2z1 = fx1 * fy2 * fz1 * m_img[mvf+moving->dim[0]];
			    m_x2y2z1 = fx2 * fy2 * fz1 * m_img[mvf+moving->dim[0]+1];
			    m_x1y1z2 = fx1 * fy1 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]];
			    m_x2y1z2 = fx2 * fy1 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+1];
			    m_x1y2z2 = fx1 * fy2 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]];
			    m_x2y2z2 = fx2 * fy2 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]+1];
			    m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 
				    + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

			    /* Compute intensity difference */
			    diff = f_img[fv] - m_val;

			    /* We'll go ahead and accumulate the score here, but you would 
			       have to reduce somewhere else instead */
			    tile_score += diff * diff;
			    tile_num_vox ++;

			    /* Compute spatial gradient using nearest neighbors */
			    mvr = (mkr * moving->dim[1] + mjr) * moving->dim[0] + mir;

			    /* Store dc_dv for this offset */
			    dc_dv[3*qidx+0] = diff * m_grad[3*mvr+0];  /* x component */
			    dc_dv[3*qidx+1] = diff * m_grad[3*mvr+1];  /* y component */
			    dc_dv[3*qidx+2] = diff * m_grad[3*mvr+2];  /* z component */
			}
		    }
		}

		//logfile_printf ("Kernel 2, tile %d %d %d\n", p[0], p[1], p[2]);
		num_vox += tile_num_vox;
		ssd->score += tile_score;

		/* Parallel across 64 control points */
#pragma omp parallel for
		for (k = 0; k < 4; k++) {
		    int i, j;
		    for (j = 0; j < 4; j++) {
			for (i = 0; i < 4; i++) {
			    int qidx;
			    int cidx;
			    int q[3];
			    int m;

			    /* Compute linear index of control point */
			    m = k*16 + j*4 + i;

			    /* Find index of control point within coefficient array */
			    cidx = c_lut[m] * 3;

			    /* Serial across offsets within kernel */
			    for (qidx = 0, q[2] = 0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
				for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
				    for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++, qidx++) {

					/* Find q_lut row for this offset */
					float* q_lut = &bxf->q_lut[qidx*64];

					/* Accumulate update to gradient for this 
					   control point */
					ssd->grad[cidx+0] += dc_dv[3*qidx+0] * q_lut[m];
					ssd->grad[cidx+1] += dc_dv[3*qidx+1] * q_lut[m];
					ssd->grad[cidx+2] += dc_dv[3*qidx+2] * q_lut[m];

				    }
				}
			    }
			}
		    }
		}
	    }
	}
    }
    free (dc_dv);

    if (parms->debug) {
	fclose (fp);
    }

    /* Normalize score for MSE */
    ssd->score = ssd->score / num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}

/* Mean-squared error version of implementation "C" */
/* Implementation "C" is slower than "B", but yields a smoother cost function 
   for use by L-BFGS-B.  It uses linear interpolation of moving image, 
   and nearest neighbor interpolation of gradient */
void
bspline_score_c_mse (
    BSPLINE_Parms *parms, 
    Bspline_state *bst,
    BSPLINE_Xform* bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad
)
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i;
    int rijk[3];             /* Indices within fixed image region (vox) */
    int fijk[3], fv;         /* Indices within fixed image (vox) */
    float mijk[3];           /* Indices within moving image (vox) */
    float fxyz[3];           /* Position within fixed image (mm) */
    float mxyz[3];           /* Position within moving image (mm) */
    int mijk_f[3], mvf;      /* Floor */
    int mijk_r[3], mvr;      /* Round */
    int p[3];
    int q[3];
    float diff;
    float dc_dv[3];
    float li_1[3];           /* Fraction of interpolant in lower index */
    float li_2[3];           /* Fraction of interpolant in upper index */
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    int num_vox;
    int pidx, qidx;
    Timer timer;
    double interval;
    float m_val;

    /* GCS: Oct 5, 2009.  We have determined that sequential accumulation
       of the score requires double precision.  However, reduction 
       accumulation does not. */
    double score_acc = 0.;

    static int it = 0;
    char debug_fn[1024];
    FILE* fp;

    if (parms->debug) {
	sprintf (debug_fn, "dc_dv_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    plm_timer_start (&timer);

    ssd->score = 0.0f;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;
    for (rijk[2] = 0, fijk[2] = bxf->roi_offset[2]; rijk[2] < bxf->roi_dim[2]; rijk[2]++, fijk[2]++) {
	p[2] = rijk[2] / bxf->vox_per_rgn[2];
	q[2] = rijk[2] % bxf->vox_per_rgn[2];
	fxyz[2] = bxf->img_origin[2] + bxf->img_spacing[2] * fijk[2];
	for (rijk[1] = 0, fijk[1] = bxf->roi_offset[1]; rijk[1] < bxf->roi_dim[1]; rijk[1]++, fijk[1]++) {
	    p[1] = rijk[1] / bxf->vox_per_rgn[1];
	    q[1] = rijk[1] % bxf->vox_per_rgn[1];
	    fxyz[1] = bxf->img_origin[1] + bxf->img_spacing[1] * fijk[1];
	    for (rijk[0] = 0, fijk[0] = bxf->roi_offset[0]; rijk[0] < bxf->roi_dim[0]; rijk[0]++, fijk[0]++) {
		int rc;
		p[0] = rijk[0] / bxf->vox_per_rgn[0];
		q[0] = rijk[0] % bxf->vox_per_rgn[0];
		fxyz[0] = bxf->img_origin[0] + bxf->img_spacing[0] * fijk[0];

		/* Get B-spline deformation vector */
		pidx = INDEX_OF (p, bxf->rdims);
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
		bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

		/* Compute moving image coordinate of fixed image voxel */
		rc = bspline_find_correspondence (mxyz, mijk, fxyz, 
						  dxyz, moving);

		/* If voxel is not inside moving image */
		if (!rc) continue;

		/* Compute interpolation fractions */
		CLAMP_LINEAR_INTERPOLATE_3D (mijk, mijk_f, mijk_r, 
					     li_1, li_2, moving);

		/* Find linear index of "corner voxel" in moving image */
		mvf = INDEX_OF (mijk_f, moving->dim);

		/* Compute moving image intensity using linear interpolation */
		/* Macro is slightly faster than function */
		BSPLINE_LI_VALUE (m_val, 
				  li_1[0], li_2[0],
				  li_1[1], li_2[1],
				  li_1[2], li_2[2],
				  mvf, m_img, moving);

		/* Compute linear index of fixed image voxel */
		fv = INDEX_OF (fijk, fixed->dim);

		/* Compute intensity difference */
		diff = f_img[fv] - m_val;

		/* Compute spatial gradient using nearest neighbors */
		mvr = INDEX_OF (mijk_r, moving->dim);
		dc_dv[0] = diff * m_grad[3*mvr+0];  /* x component */
		dc_dv[1] = diff * m_grad[3*mvr+1];  /* y component */
		dc_dv[2] = diff * m_grad[3*mvr+2];  /* z component */
		bspline_update_grad_b_inline (bst, bxf, pidx, qidx, dc_dv);
		
		if (parms->debug) {
		    fprintf (fp, "%d %d %d %g %g %g\n", 
			     rijk[0], rijk[1], rijk[2], 
			     dc_dv[0], dc_dv[1], dc_dv[2]);
		}
		score_acc += diff * diff;
		num_vox ++;
	    }
	}
    }

    if (parms->debug) {
	fclose (fp);
    }

    /* Normalize score for MSE */
    ssd->score = score_acc / num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}

/* This is the fastest known version.  It does nearest neighbors 
   interpolation of both moving image and gradient which doesn't 
   work with stock L-BFGS-B optimizer. */
void
bspline_score_b_mse 
(
 BSPLINE_Parms *parms, 
 Bspline_state *bst,
 BSPLINE_Xform *bxf, 
 Volume *fixed, 
 Volume *moving, 
 Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i;
    int ri, rj, rk;
    int fi, fj, fk, fv;
    int mi, mj, mk, mv;
    float fx, fy, fz;
    float mx, my, mz;
    int p[3];
    int q[3];
    float diff;
    float dc_dv[3];
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    int num_vox;
    int pidx, qidx;
    Timer timer;
    double interval;

    plm_timer_start (&timer);

    ssd->score = 0;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;
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
		pidx = INDEX_OF (p, bxf->rdims);
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
		bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

		/* Compute coordinate of fixed image voxel */
		fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

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
		mv = (mk * moving->dim[1] + mj) * moving->dim[0] + mi;

		/* Compute intensity difference */
		diff = f_img[fv] - m_img[mv];

		/* Compute spatial gradient using nearest neighbors */
		dc_dv[0] = diff * m_grad[3*mv+0];  /* x component */
		dc_dv[1] = diff * m_grad[3*mv+1];  /* y component */
		dc_dv[2] = diff * m_grad[3*mv+2];  /* z component */

		bspline_update_grad_b (bst, bxf, pidx, qidx, dc_dv);
		
		ssd->score += diff * diff;
		num_vox ++;
	    }
	}
    }

    /* Normalize score for MSE */
    ssd->score /= num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}



void
bspline_score_a_mse 
(
 BSPLINE_Parms *parms, 
 Bspline_state *bst,
 BSPLINE_Xform* bxf, 
 Volume *fixed, 
 Volume *moving, 
 Volume *moving_grad
 )
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i;
    int ri, rj, rk;
    int fi, fj, fk, fv;
    int mi, mj, mk, mv;
    float fx, fy, fz;
    float mx, my, mz;
    int p[3];
    int q[3];
    float diff;
    float dc_dv[3];
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    int num_vox;
    int qidx;
    Timer timer;
    double interval;

    plm_timer_start (&timer);

    ssd->score = 0;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;
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

		/* Compute coordinate of fixed image voxel */
		fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

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
		mv = (mk * moving->dim[1] + mj) * moving->dim[0] + mi;

		/* Compute intensity difference */
		diff = f_img[fv] - m_img[mv];

		/* Compute spatial gradient using nearest neighbors */
		dc_dv[0] = diff * m_grad[3*mv+0];  /* x component */
		dc_dv[1] = diff * m_grad[3*mv+1];  /* y component */
		dc_dv[2] = diff * m_grad[3*mv+2];  /* z component */
		bspline_update_grad (bst, bxf, p, qidx, dc_dv);
		
		ssd->score += diff * diff;
		num_vox ++;
	    }
	}
    }

    /* Normalize score for MSE */
    ssd->score /= num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] /= num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}
