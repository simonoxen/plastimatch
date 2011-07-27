/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "bspline.h"
#include "bspline_xform.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "reg.h"
#include "volume.h"
#include "plm_timer.h"

//#define DEBUG

void
print_matrix (double* mat, int m, int n)
{
    int i,j;

    for (j=0; j<n; j++) {
        for (i=0; i<m; i++) {
            printf ("%1.3e ", mat[m*j+i]);
        }
        printf ("\n");
    }
}

Volume*
compute_vf_from_coeff (const Bspline_xform* bxf)
{
    Volume* vf = new Volume (
	bxf->img_dim, bxf->img_origin, 
	bxf->img_spacing, 0, 
	PT_VF_FLOAT_INTERLEAVED, 3, 0
    );
    bspline_interpolate_vf (vf, bxf);

    return vf;
}

void
compute_coeff_from_vf (Bspline_xform* bxf, Volume* vol)
{
    int i,j,k;
    int a,b,c,z;
    int idx_poi, cidx, pidx, qidx;
    float *vec_poi;
    float *img = (float*) vol->img;

    int p[3];
    float q[3];
    float* q_lut;
    int* c_lut;

    for (k = 0; k < vol->dim[2]; k++) {
        p[2] = k / bxf->vox_per_rgn[2];
        q[2] = k % bxf->vox_per_rgn[2];
        for (j = 0; j < vol->dim[1]; j++) {
            p[1] = j / bxf->vox_per_rgn[1];
            q[1] = j % bxf->vox_per_rgn[1];
            for (i = 0; i < vol->dim[0]; i++) {
                p[0] = i / bxf->vox_per_rgn[0];
                q[0] = i % bxf->vox_per_rgn[0];

                pidx = volume_index (bxf->rdims, p[0], p[1], p[2]);
                qidx = volume_index (bxf->vox_per_rgn, q[0], q[1], q[2]);

                idx_poi = volume_index (vol->dim, i, j, k);
                vec_poi = &img[3*idx_poi];

                q_lut = &bxf->q_lut[qidx*64];
                c_lut = &bxf->c_lut[pidx*64];

                z = 0;
                for (c = 0; c < 4; c++) {
                    for (b = 0; b < 4; b++) {
                        for (a = 0; a < 4; a++) {
                            cidx = 3 * c_lut[z];
                            bxf->coeff[cidx+0] += vec_poi[0] * q_lut[z];
                            bxf->coeff[cidx+1] += vec_poi[1] * q_lut[z];
                            bxf->coeff[cidx+2] += vec_poi[2] * q_lut[z];
                            z++;
                        }
                    }
                }

            } /* i < vol-dim[0] */
        } /* j < vol->dim[1] */
    } /* k < vol->dim[2] */
}


#if (OPENMP_FOUND)
void
reg_sort_sets (
    double* cond,
    double* sets,
    int* k_lut,
    const Bspline_xform* bxf
)
{
    int sidx, kidx;

    /* Rackem' Up */
    for (sidx=0; sidx<64; sidx++) {
        kidx = k_lut[sidx];

        cond[3*(64*kidx+sidx)+0] = sets[3*sidx+0];
        cond[3*(64*kidx+sidx)+1] = sets[3*sidx+1];
        cond[3*(64*kidx+sidx)+2] = sets[3*sidx+2];
    }
}
#endif


#if (OPENMP_FOUND)
void
reg_update_grad (
    Bspline_score* ssd,
    double* cond,
    const Bspline_xform* bxf
)
{
    int kidx, sidx;

    for (kidx=0; kidx < (bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2]); kidx++) {
        for (sidx=0; sidx<64; sidx++) {
            ssd->grad[3*kidx+0] += cond[3*(64*kidx+sidx)+0];
            ssd->grad[3*kidx+1] += cond[3*(64*kidx+sidx)+1];
            ssd->grad[3*kidx+2] += cond[3*(64*kidx+sidx)+2];
        }
    }
}
#endif


void 
find_knots (int* knots, int tile_num, const int* cdims)
{
    int tile_loc[3];
    int i, j, k;
    int idx = 0;
    int num_tiles_x = cdims[0] - 3;
    int num_tiles_y = cdims[1] - 3;
    int num_tiles_z = cdims[2] - 3;

    // First get the [x,y,z] coordinate of
    // the tile in the control grid.
    tile_loc[0] = tile_num % num_tiles_x;
    tile_loc[1] = ((tile_num - tile_loc[0]) / num_tiles_x) % num_tiles_y;
    tile_loc[2] = ((((tile_num - tile_loc[0]) / num_tiles_x) / num_tiles_y) % num_tiles_z);

    /* GCS 2011-07-14: Why not remove the below three lines, and let i,j,k 
       run from 0 to 3? */
    // Tiles do not start on the edges of the grid, so we
    // push them to the center of the control grid.
    tile_loc[0]++;
    tile_loc[1]++;
    tile_loc[2]++;

    // Find 64 knots' [x,y,z] coordinates
    // and convert into a linear knot index
    for (k = -1; k < 3; k++)
    	for (j = -1; j < 3; j++)
    	    for (i = -1; i < 3; i++) {
    		knots[idx++] = (cdims[0]*cdims[1]*(tile_loc[2]+k)) +
                           (cdims[0]*(tile_loc[1]+j)) +
                           (tile_loc[0]+i);
        }

}

void
eval_integral (double* V, double* Qn, double gs)
{
    int i,j;
    double S[16];

    double I[7] = {
        gs,
        (1.0/2.0) * (gs * gs),
        (1.0/3.0) * (gs * gs * gs),
        (1.0/4.0) * (gs * gs * gs * gs),
        (1.0/5.0) * (gs * gs * gs * gs * gs),
        (1.0/6.0) * (gs * gs * gs * gs * gs * gs),
        (1.0/7.0) * (gs * gs * gs * gs * gs * gs * gs)
    };

    // Generate 4 4x4 matrix by taking the outer
    // product of the each row in the Q matrix with
    // every other row in the Q matrix. We use these
    // to generate each element in V.
    for (j=0; j<4; j++) {
        for (i=0; i<4; i++) {
            vec_outer (S, Qn+(4*j), Qn+(4*i), 4);
            V[4*j + i] = (I[0] *  S[ 0])
                       + (I[1] * (S[ 1] + S[ 4]))
                       + (I[2] * (S[ 2] + S[ 5] + S[ 8]))
                       + (I[3] * (S[ 3] + S[ 6] + S[ 9] + S[12]))
                       + (I[4] * (S[ 7] + S[10] + S[13]))
                       + (I[5] * (S[11] + S[14]))
                       + (I[6] * (S[15]));
        }
    }
}

void
init_analytic (double **QX, double **QY, double **QZ, 
    const Bspline_xform* bxf)
{
    double rx, ry, rz;

    double B[16] = {
        1.0/6.0, -1.0/2.0,  1.0/2.0, -1.0/6.0,
        2.0/3.0,  0.0    , -1.0    ,  1.0/2.0,
        1.0/6.0,  1.0/2.0,  1.0/2.0, -1.0/2.0,
        0.0    ,  0.0    ,  0.0    ,  1.0/6.0
    };

    /* grid spacing */
    rx = 1.0/bxf->grid_spac[0];
    ry = 1.0/bxf->grid_spac[1];
    rz = 1.0/bxf->grid_spac[2];

    double RX[16] = {
        1.0, 0.0,   0.0,      0.0,
        0.0,  rx,   0.0,      0.0,
        0.0, 0.0, rx*rx,      0.0,
        0.0, 0.0,   0.0, rx*rx*rx
    };

    double RY[16] = {
        1.0, 0.0,   0.0,      0.0,
        0.0,  ry,   0.0,      0.0,
        0.0, 0.0, ry*ry,      0.0,
        0.0, 0.0,   0.0, ry*ry*ry
    };

    double RZ[16] = {
        1.0, 0.0,   0.0,      0.0,
        0.0,  rz,   0.0,      0.0,
        0.0, 0.0, rz*rz,      0.0,
        0.0, 0.0,   0.0, rz*rz*rz
    };

    double delta1[16] = {
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0
    };

    double delta2[16] = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        2.0, 0.0, 0.0, 0.0,
        0.0, 6.0, 0.0, 0.0
    };

    // Let's call Q the product of the recripocal grid spacing
    // matrix (R) and the B-spline coefficient matrix (B).
    mat_mult_mat (QX[0], B, 4, 4, RX, 4, 4);
    mat_mult_mat (QY[0], B, 4, 4, RY, 4, 4);
    mat_mult_mat (QZ[0], B, 4, 4, RZ, 4, 4);

    // Get the product of QX, QY, QZ and delta.
    //   QX1 is the  first-order derivative of X
    //   QX2 is the second-order derivative of X
    //   QY1 is the  first-order derivative of Y
    //   ... etc
    mat_mult_mat (QX[1], QX[0], 4, 4, delta1, 4, 4);    
    mat_mult_mat (QX[2], QX[0], 4, 4, delta2, 4, 4);    
    mat_mult_mat (QY[1], QY[0], 4, 4, delta1, 4, 4);    
    mat_mult_mat (QY[2], QY[0], 4, 4, delta2, 4, 4);    
    mat_mult_mat (QZ[1], QZ[0], 4, 4, delta1, 4, 4);    
    mat_mult_mat (QZ[2], QZ[0], 4, 4, delta2, 4, 4);    
}

void
get_Vmatrix (double* V, double* X, double* Y, double* Z)
{
    int i,j;
    double tmp[256];       /* 16 x 16 matrix */

    /* Calculate the temporary 16*16 matrix */
    for (j=0; j<4; j++) {
        for (i=0; i<4; i++) {
            tmp[16*(j+ 0) + (i+ 0)] = Y[4*0 + 0] * Z[4*j + i];
            tmp[16*(j+ 0) + (i+ 4)] = Y[4*0 + 1] * Z[4*j + i];
            tmp[16*(j+ 0) + (i+ 8)] = Y[4*0 + 2] * Z[4*j + i];
            tmp[16*(j+ 0) + (i+12)] = Y[4*0 + 3] * Z[4*j + i];

            tmp[16*(j+ 4) + (i+ 0)] = Y[4*1 + 0] * Z[4*j + i];
            tmp[16*(j+ 4) + (i+ 4)] = Y[4*1 + 1] * Z[4*j + i];
            tmp[16*(j+ 4) + (i+ 8)] = Y[4*1 + 2] * Z[4*j + i];
            tmp[16*(j+ 4) + (i+12)] = Y[4*1 + 3] * Z[4*j + i];

            tmp[16*(j+ 8) + (i+ 0)] = Y[4*2 + 0] * Z[4*j + i];
            tmp[16*(j+ 8) + (i+ 4)] = Y[4*2 + 1] * Z[4*j + i];
            tmp[16*(j+ 8) + (i+ 8)] = Y[4*2 + 2] * Z[4*j + i];
            tmp[16*(j+ 8) + (i+12)] = Y[4*2 + 3] * Z[4*j + i];

            tmp[16*(j+12) + (i+ 0)] = Y[4*3 + 0] * Z[4*j + i];
            tmp[16*(j+12) + (i+ 4)] = Y[4*3 + 1] * Z[4*j + i];
            tmp[16*(j+12) + (i+ 8)] = Y[4*3 + 2] * Z[4*j + i];
            tmp[16*(j+12) + (i+12)] = Y[4*3 + 3] * Z[4*j + i];
        }
    }

    /* Calculate the 64*64 V matrix */
    for (j=0; j<16; j++) {
        for (i=0; i<16; i++) {
            V[64*(j+ 0) + (i+ 0)] = X[4*0 + 0] * tmp[16*j + i];
            V[64*(j+ 0) + (i+16)] = X[4*0 + 1] * tmp[16*j + i];
            V[64*(j+ 0) + (i+32)] = X[4*0 + 2] * tmp[16*j + i];
            V[64*(j+ 0) + (i+48)] = X[4*0 + 3] * tmp[16*j + i];

            V[64*(j+16) + (i+ 0)] = X[4*1 + 0] * tmp[16*j + i];
            V[64*(j+16) + (i+16)] = X[4*1 + 1] * tmp[16*j + i];
            V[64*(j+16) + (i+32)] = X[4*1 + 2] * tmp[16*j + i];
            V[64*(j+16) + (i+48)] = X[4*1 + 3] * tmp[16*j + i];

            V[64*(j+32) + (i+ 0)] = X[4*2 + 0] * tmp[16*j + i];
            V[64*(j+32) + (i+16)] = X[4*2 + 1] * tmp[16*j + i];
            V[64*(j+32) + (i+32)] = X[4*2 + 2] * tmp[16*j + i];
            V[64*(j+32) + (i+48)] = X[4*2 + 3] * tmp[16*j + i];

            V[64*(j+48) + (i+ 0)] = X[4*3 + 0] * tmp[16*j + i];
            V[64*(j+48) + (i+16)] = X[4*3 + 1] * tmp[16*j + i];
            V[64*(j+48) + (i+32)] = X[4*3 + 2] * tmp[16*j + i];
            V[64*(j+48) + (i+48)] = X[4*3 + 3] * tmp[16*j + i];
        }
    }
}

/* Employs my "world famous" thread safe "condense" control-point
 * update method for mulit-core acceleration.  Also, score is
 * returned so as to take advantage of OpenMP's built in
 * sum-reduction capabilities */
#if (OPENMP_FOUND)
double
region_smoothness_omp (
    double* sets,
    const Reg_parms* reg_parms,    
    const Bspline_xform* bxf,
    double* V, 
    int* knots
)
{
    double S = 0.0;         /* Region smoothness */
    double X[64] = {0};
    double Y[64] = {0};
    double Z[64] = {0};
    int i,j;

    for (j=0; j<64; j++) {
    	/* S = pVp operation ----------------------------- */
        for (i=0; i<64; i++) {
            X[j] += bxf->coeff[3*knots[i]+0] * V[64*j + i];
            Y[j] += bxf->coeff[3*knots[i]+1] * V[64*j + i];
            Z[j] += bxf->coeff[3*knots[i]+2] * V[64*j + i];
        }

        S += X[j] * bxf->coeff[3*knots[j]+0];
        S += Y[j] * bxf->coeff[3*knots[j]+1];
        S += Z[j] * bxf->coeff[3*knots[j]+2];
        /* ------------------------------------------------ */

        /* dS/dp = 2Vp operation */
        sets[3*j+0] += 2 * reg_parms->lambda * X[j];
        sets[3*j+1] += 2 * reg_parms->lambda * Y[j];
        sets[3*j+2] += 2 * reg_parms->lambda * Z[j];
    }

    return S;
}
#endif

void
region_smoothness (
    Bspline_score *bspline_score, 
    const Reg_parms* reg_parms,    
    const Bspline_xform* bxf,
    double* V, 
    int* knots)
{
    double S = 0.0;         /* Region smoothness */
    double X[64] = {0};
    double Y[64] = {0};
    double Z[64] = {0};
    int i,j;

    for (j=0; j<64; j++) {
    	/* S = pVp operation ----------------------------- */
        for (i=0; i<64; i++) {
            X[j] += bxf->coeff[3*knots[i]+0] * V[64*j + i];
            Y[j] += bxf->coeff[3*knots[i]+1] * V[64*j + i];
            Z[j] += bxf->coeff[3*knots[i]+2] * V[64*j + i];
        }

        S += X[j] * bxf->coeff[3*knots[j]+0];
        S += Y[j] * bxf->coeff[3*knots[j]+1];
        S += Z[j] * bxf->coeff[3*knots[j]+2];
        /* ------------------------------------------------ */

        /* dS/dp = 2Vp operation */
        bspline_score->grad[3*knots[j]+0] += 2 * reg_parms->lambda * X[j];
        bspline_score->grad[3*knots[j]+1] += 2 * reg_parms->lambda * Y[j];
        bspline_score->grad[3*knots[j]+2] += 2 * reg_parms->lambda * Z[j];
    }

    bspline_score->rmetric += sqrt(S*S);
}


void
vf_regularize_analytic_init (
    Reg_state* rst,
    const Bspline_xform* bxf)
{
    double X[256];                      /* 16 x 16 matrix */
    double Y[256];                      /* 16 x 16 matrix */
    double Z[256];                      /* 16 x 16 matrix */
    double gs[3];

    rst->cond = (double*)malloc(3*64*bxf->num_knots*sizeof(double));

    gs[0] = (double)bxf->grid_spac[0];
    gs[1] = (double)bxf->grid_spac[1];
    gs[2] = (double)bxf->grid_spac[2];

    rst->QX_mats = (double*)malloc (3 * 16 * sizeof (double));
    rst->QY_mats = (double*)malloc (3 * 16 * sizeof (double));
    rst->QZ_mats = (double*)malloc (3 * 16 * sizeof (double));

    memset (rst->QX_mats, 0, 3*16*sizeof(double));
    memset (rst->QY_mats, 0, 3*16*sizeof(double));
    memset (rst->QZ_mats, 0, 3*16*sizeof(double));

    rst->QX = (double**)malloc (3 * sizeof (double*));
    rst->QY = (double**)malloc (3 * sizeof (double*));
    rst->QZ = (double**)malloc (3 * sizeof (double*));

    /* 4x4 matrices */
    rst->QX[0] = rst->QX_mats;
    rst->QX[1] = rst->QX[0] + 16;
    rst->QX[2] = rst->QX[1] + 16;

    rst->QY[0] = rst->QY_mats;
    rst->QY[1] = rst->QY[0] + 16;
    rst->QY[2] = rst->QY[1] + 16;

    rst->QZ[0] = rst->QZ_mats;
    rst->QZ[1] = rst->QZ[0] + 16;
    rst->QZ[2] = rst->QZ[1] + 16;

    init_analytic (rst->QX, rst->QY, rst->QZ, bxf);

    /* The below should probably be wrapped into init_analytic() */
    rst->V_mats = (double*)malloc (6*4096 * sizeof (double));
    rst->V = (double**)malloc (6 * sizeof (double*));

    /* The six 64 x 64 V matrices */
    rst->V[0] = rst->V_mats;
    rst->V[1] = rst->V[0] + 4096;
    rst->V[2] = rst->V[1] + 4096;
    rst->V[3] = rst->V[2] + 4096;
    rst->V[4] = rst->V[3] + 4096;
    rst->V[5] = rst->V[4] + 4096;

    eval_integral (X, rst->QX[2], gs[0]);
    eval_integral (Y, rst->QY[0], gs[1]);
    eval_integral (Z, rst->QZ[0], gs[2]);
    get_Vmatrix (rst->V[0], X, Y, Z);

    eval_integral (X, rst->QX[0], gs[0]);
    eval_integral (Y, rst->QY[2], gs[1]);
    eval_integral (Z, rst->QZ[0], gs[2]);
    get_Vmatrix (rst->V[1], X, Y, Z);

    eval_integral (X, rst->QX[0], gs[0]);
    eval_integral (Y, rst->QY[0], gs[1]);
    eval_integral (Z, rst->QZ[2], gs[2]);
    get_Vmatrix (rst->V[2], X, Y, Z);

    eval_integral (X, rst->QX[1], gs[0]);
    eval_integral (Y, rst->QY[1], gs[1]);
    eval_integral (Z, rst->QZ[0], gs[2]);
    get_Vmatrix (rst->V[3], X, Y, Z);

    eval_integral (X, rst->QX[1], gs[0]);
    eval_integral (Y, rst->QY[0], gs[1]);
    eval_integral (Z, rst->QZ[1], gs[2]);
    get_Vmatrix (rst->V[4], X, Y, Z);

    eval_integral (X, rst->QX[0], gs[0]);
    eval_integral (Y, rst->QY[1], gs[1]);
    eval_integral (Z, rst->QZ[1], gs[2]);
    get_Vmatrix (rst->V[5], X, Y, Z);

    printf ("Regularizer initialized\n");
}

void
vf_regularize_analytic_destroy (
    Reg_state* rst
)
{
    free (rst->cond);

    free (rst->QX);
    free (rst->QY);
    free (rst->QZ);

    free (rst->QX_mats);
    free (rst->QY_mats);
    free (rst->QZ_mats);

    free (rst->V_mats);
    free (rst->V);
}


/* flavor 'c' */
#if (OPENMP_FOUND)
void
vf_regularize_analytic_omp (
    Bspline_score *bspline_score, 
    const Reg_parms* reg_parms,
    const Reg_state* rst,
    const Bspline_xform* bxf)
{
    int i,n;
    Timer timer;

    double S = 0.0;

    plm_timer_start (&timer);

    memset (rst->cond, 0, 3*64*bxf->num_knots * sizeof (double));

    // Total number of regions in grid
    n = bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2];

    bspline_score->rmetric = 0.0;

#pragma omp parallel for reduction(+:S)
    for (i=0; i<n; i++) {
        int knots[64];
        double sets[3*64];

        memset (sets, 0, 3*64*sizeof (double));

        find_knots (knots, i, bxf->cdims);

        S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[0], knots);
        S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[1], knots);
        S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[2], knots);
        S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[3], knots);
        S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[4], knots);
        S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[5], knots);

        reg_sort_sets (rst->cond, sets, knots, bxf);
    }

    reg_update_grad (bspline_score, rst->cond, bxf);

    bspline_score->rmetric = S;
    bspline_score->time_rmetric = plm_timer_report (&timer);
}
#endif


/* flavor 'b' */
void
vf_regularize_analytic (
    Bspline_score *bspline_score, 
    const Reg_parms* reg_parms,
    const Reg_state* rst,
    const Bspline_xform* bxf)
{
    int i,n;
    int knots[64];
    Timer timer;

    plm_timer_start (&timer);

    // Total number of regions in grid
    n = bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2];

    bspline_score->rmetric = 0.0;

    for (i=0; i<n; i++) {
        // Get the set of 64 control points for this region
        find_knots (knots, i, bxf->cdims);

        region_smoothness (bspline_score, reg_parms, bxf, rst->V[0], knots);
        region_smoothness (bspline_score, reg_parms, bxf, rst->V[1], knots);
        region_smoothness (bspline_score, reg_parms, bxf, rst->V[2], knots);
        region_smoothness (bspline_score, reg_parms, bxf, rst->V[3], knots);
        region_smoothness (bspline_score, reg_parms, bxf, rst->V[4], knots);
        region_smoothness (bspline_score, reg_parms, bxf, rst->V[5], knots);
    }

    bspline_score->time_rmetric = plm_timer_report (&timer);
}

/* flavor 'a' */
float
vf_regularize_numerical (Volume* vol)
{
#if defined (DEBUG)
    FILE* fp[3];
#endif

    int i,j,k,c;
    float *img = (float*) vol->img;

    float dx = vol->spacing[0];
    float dy = vol->spacing[1];
    float dz = vol->spacing[2];

    float inv_dxdx = 1.0f / (dx * dx);
    float inv_dydy = 1.0f / (dy * dy);
    float inv_dzdz = 1.0f / (dz * dz);

    float inv_dxdy = 0.25f / (dx*dy);
    float inv_dxdz = 0.25f / (dx*dz);
    float inv_dydz = 0.25f / (dy*dz);


    /* Index of current point-of-interest (POI) */
    int idx_poi;

    /* Indices of POI's SxS neighbors */
    int idx_in, idx_ip;
    int idx_jn, idx_jp;
    int idx_kn, idx_kp;

    /* Indicies of POI's diagonal neighbors */
    int idx_injn, idx_injp, idx_ipjn, idx_ipjp;
    int idx_inkn, idx_inkp, idx_ipkn, idx_ipkp;
    int idx_jnkn, idx_jnkp, idx_jpkn, idx_jpkp;

    /* Deformation vector @ POI */
    float *vec_poi;

    /* Vectors of POI's SxS neighbors */
    float *vec_in, *vec_ip;
    float *vec_jn, *vec_jp;
    float *vec_kn, *vec_kp;

    /* Vectors of POI's diagonal neighbors */
    float *vec_injn, *vec_injp;
    float *vec_ipjn, *vec_ipjp;
    float *vec_inkn, *vec_inkp;
    float *vec_ipkn, *vec_ipkp;
    float *vec_jnkn, *vec_jnkp;
    float *vec_jpkn, *vec_jpkp;

    /* Vector's partial spatial derivatives */
    float d2_dx2[3],  d2_dxdy[3];
    float d2_dy2[3],  d2_dxdz[3];
    float d2_dz2[3],  d2_dydz[3];

    /* Square of 2nd derivative */
    float d2_sq, dd2_dxdy;

    /* Smoothness */
    float S, SS;

#if defined (DEBUG)
    printf ("Warning: compiled with DEBUG : writing to to files:\n");
    printf ("  d2ux_dxy_sq.txt\n"); fp[0] = fopen ("d2ux_dxdy_sq.txt", "w");
    printf ("  d2uy_dxy_sq.txt\n"); fp[1] = fopen ("d2uy_dxdy_sq.txt", "w");
    printf ("  d2uz_dxy_sq.txt\n"); fp[2] = fopen ("d2uz_dxdy_sq.txt", "w");
#endif

    S = 0.0f, SS=0.0f;
    for (k = 1; k < vol->dim[2]-1; k++) {
        for (j = 1; j < vol->dim[1]-1; j++) {
            for (i = 1; i < vol->dim[0]-1; i++) {

                /* Load indicies relevant to current POI */
                idx_poi = volume_index (vol->dim, i, j, k);

                idx_in = volume_index (vol->dim, i-1  , j,   k);
                idx_ip = volume_index (vol->dim, i+1,   j,   k);
                idx_jn = volume_index (vol->dim,   i, j-1,   k);
                idx_jp = volume_index (vol->dim,   i, j+1,   k);
                idx_kn = volume_index (vol->dim,   i,   j, k-1);
                idx_kp = volume_index (vol->dim,   i,   j, k+1);

                idx_injn = volume_index (vol->dim, i-1, j-1,   k);
                idx_injp = volume_index (vol->dim, i-1, j+1,   k);
                idx_ipjn = volume_index (vol->dim, i+1, j-1,   k);
                idx_ipjp = volume_index (vol->dim, i+1, j+1,   k);
                idx_inkn = volume_index (vol->dim, i-1,   j, k-1);
                idx_inkp = volume_index (vol->dim, i-1,   j, k+1);
                idx_ipkn = volume_index (vol->dim, i+1,   j, k-1);
                idx_ipkp = volume_index (vol->dim, i+1,   j, k+1);
                idx_jnkn = volume_index (vol->dim,   i, j-1, k-1);
                idx_jnkp = volume_index (vol->dim,   i, j-1, k+1);
                idx_jpkn = volume_index (vol->dim,   i, j+1, k-1);
                idx_jpkp = volume_index (vol->dim,   i, j+1, k+1);

                /* Load vectors relevant to current POI */
                vec_poi = &img[3*idx_poi];

                vec_in = &img[3*idx_in]; vec_ip = &img[3*idx_ip];
                vec_jn = &img[3*idx_jn]; vec_jp = &img[3*idx_jp];
                vec_kn = &img[3*idx_kn]; vec_kp = &img[3*idx_kp];

                vec_injn = &img[3*idx_injn]; vec_injp = &img[3*idx_injp];
                vec_ipjn = &img[3*idx_ipjn]; vec_ipjp = &img[3*idx_ipjp];
                vec_inkn = &img[3*idx_inkn]; vec_inkp = &img[3*idx_inkp];
                vec_ipkn = &img[3*idx_ipkn]; vec_ipkp = &img[3*idx_ipkp];
                vec_jnkn = &img[3*idx_jnkn]; vec_jnkp = &img[3*idx_jnkp];
                vec_jpkn = &img[3*idx_jpkn]; vec_jpkp = &img[3*idx_jpkp];

                /* Compute components */
                d2_sq = 0.0f, dd2_dxdy=0.0f;
                for (c=0; c<3; c++) {
                    d2_dx2[c] = inv_dxdx * (vec_ip[c] - 2.0f*vec_poi[c] + vec_in[c]);
                    d2_dy2[c] = inv_dydy * (vec_jp[c] - 2.0f*vec_poi[c] + vec_jn[c]);
                    d2_dz2[c] = inv_dzdz * (vec_kp[c] - 2.0f*vec_poi[c] + vec_kn[c]);

                    d2_dxdy[c] = inv_dxdy * (
                        vec_injn[c] - vec_injp[c] - vec_ipjn[c] + vec_ipjp[c]);
                    d2_dxdz[c] = inv_dxdz * (
                        vec_inkn[c] - vec_inkp[c] - vec_ipkn[c] + vec_ipkp[c]);
                    d2_dydz[c] = inv_dydz * (
                        vec_jnkn[c] - vec_jnkp[c] - vec_jpkn[c] + vec_jpkp[c]);

                    d2_sq += d2_dx2[c]*d2_dx2[c] + d2_dy2[c]*d2_dy2[c] +
                             d2_dz2[c]*d2_dz2[c] + 2.0f * (
                                d2_dxdy[c]*d2_dxdy[c] +
                                d2_dxdz[c]*d2_dxdz[c] +
                                d2_dydz[c]*d2_dydz[c]
                        );
					

#if defined (DEBUG)
                    fprintf (fp[c], "(%i,%i,%i) : %15e\n", i,j,k, (d2_dxdy[c]*d2_dxdy[c]));
#endif
                }

                S += d2_sq;
				
            }
        }
    }

    /* Integrate */
    S *= dx*dy*dz;

#if defined (DEBUG)
    for (i=0; i<3; i++) {
        fclose(fp[i]);
    }
#endif

    return S;
}

void
regularize (
    Bspline_score *bspline_score,    /* Gets updated */
    const Reg_state* rst,
    const Reg_parms* reg_parms,
    const Bspline_xform* bxf
)
{
    switch (reg_parms->implementation) {
    case 'a':
//        S = vf_regularize_numerical (compute_vf_from_coeff (bxf));
        print_and_exit (
            "Sorry, regularization implementation (%c) is currently unavailable.\n",
            reg_parms->implementation
        );
        break;
    case 'b':
        vf_regularize_analytic (bspline_score, reg_parms, rst, bxf);
        break;
    case 'c':
#if (OPENMP_FOUND)
        vf_regularize_analytic_omp (bspline_score, reg_parms, rst, bxf);
#else
        vf_regularize_analytic (bspline_score, reg_parms, rst, bxf);
#endif
        break;
    default:
        print_and_exit (
            "Error: unknown reg_parms->implementation (%c)\n",
            reg_parms->implementation
        );
        break;
    }
}
