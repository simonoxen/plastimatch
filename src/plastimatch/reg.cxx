/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */

#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "bspline_xform.h"
#include "volume.h"
#include "reg.h"

#define DEBUG

#define INDEX_OF(dim, i, j, k) \
    ((((k)*dim[1] + (j))*dim[0]) + (i))

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
        for (j = 0; j < vol->dim[2]; j++) {
            p[1] = j / bxf->vox_per_rgn[1];
            q[1] = j % bxf->vox_per_rgn[1];
            for (i = 0; i < vol->dim[2]; i++) {
                p[0] = i / bxf->vox_per_rgn[0];
                q[0] = i % bxf->vox_per_rgn[0];

                pidx = INDEX_OF (p, bxf->rdims[0],
                                    bxf->rdims[1],
                                    bxf->rdims[2]);
                qidx = INDEX_OF (q, bxf->vox_per_rgn[0],
                                    bxf->vox_per_rgn[1],
                                    bxf->vox_per_rgn[2]);

                idx_poi = INDEX_OF (vol->dim, i, j, k);
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

    float inv_dxdy = 0.5f / (dx*dy);
    float inv_dxdz = 0.5f / (dx*dz);
    float inv_dydz = 0.5f / (dy*dz);


    /* Index of current point-of-interest (POI) */
    int idx_poi;

    /* Indices of POI's SxS neighbors */
    int idx_in, idx_ip;
    int idx_jn, idx_jp;
    int idx_kn, idx_kp;

    /* Indicies of POI's diagonal neighbors */
    int idx_ijn, idx_ijp;
    int idx_ikn, idx_ikp;
    int idx_jkn, idx_jkp;

    /* Deformation vector @ POI */
    float *vec_poi;

    /* Vectors of POI's SxS neighbors */
    float *vec_in, *vec_ip;
    float *vec_jn, *vec_jp;
    float *vec_kn, *vec_kp;

    /* Vectors of POI's diagonal neighbors */
    float *vec_ijn, *vec_ijp;
    float *vec_ikn, *vec_ikp;
    float *vec_jkn, *vec_jkp;

    /* Vector's partial spatial derivatives */
    float d2_dx2[3],  d2_dxdy[3];
    float d2_dy2[3],  d2_dxdz[3];
    float d2_dz2[3],  d2_dydz[3];

    /* Square of 2nd derivative */
    float d2_sq;

    /* Smoothness */
    float S;

#if defined (DEBUG)
    printf ("Warning: compiled with DEBUG : writing to to files:\n");
    printf ("  d2ux_dxy_sq.txt\n"); fp[0] = fopen ("d2ux_dxdy_sq.txt", "w");
    printf ("  d2uy_dxy_sq.txt\n"); fp[1] = fopen ("d2uy_dxdy_sq.txt", "w");
    printf ("  d2uz_dxy_sq.txt\n"); fp[2] = fopen ("d2uz_dxdy_sq.txt", "w");
#endif

    S = 0.0f;
    for (k = 1; k < vol->dim[2]-1; k++) {
        for (j = 1; j < vol->dim[1]-1; j++) {
            for (i = 1; i < vol->dim[0]-1; i++) {

                /* Load indicies relevant to current POI */
                idx_poi = INDEX_OF (vol->dim, i, j, k);

                idx_in = INDEX_OF (vol->dim, i-1  , j,   k);
                idx_ip = INDEX_OF (vol->dim, i+1,   j,   k);
                idx_jn = INDEX_OF (vol->dim,   i, j-1,   k);
                idx_jp = INDEX_OF (vol->dim,   i, j+1,   k);
                idx_kn = INDEX_OF (vol->dim,   i,   j, k-1);
                idx_kp = INDEX_OF (vol->dim,   i,   j, k+1);

                idx_ijn = INDEX_OF (vol->dim, i-1, j-1,   k);
                idx_ijp = INDEX_OF (vol->dim, i+1, j+1,   k);
                idx_ikn = INDEX_OF (vol->dim, i-1,   j, k-1);
                idx_ikp = INDEX_OF (vol->dim, i+1,   j, k+1);
                idx_jkn = INDEX_OF (vol->dim,   i, j-1, k-1);
                idx_jkp = INDEX_OF (vol->dim,   i, j+1, k+1);

                /* Load vectors relevant to current POI */
                vec_poi = &img[3*idx_poi];

                vec_in = &img[3*idx_in]; vec_ip = &img[3*idx_ip];
                vec_jn = &img[3*idx_jn]; vec_jp = &img[3*idx_jp];
                vec_kn = &img[3*idx_kn]; vec_kp = &img[3*idx_kp];

                vec_ijn = &img[3*idx_ijn]; vec_ijp = &img[3*idx_ijp];
                vec_ikn = &img[3*idx_ikn]; vec_ikp = &img[3*idx_ikp];
                vec_jkn = &img[3*idx_jkn]; vec_jkp = &img[3*idx_jkp];

                /* Compute components */
                d2_sq = 0.0f;
                for (c=0; c<3; c++) {
                    d2_dx2[c] = inv_dxdx * (vec_ip[c] - 2.0f*vec_poi[c] + vec_in[c]);
                    d2_dy2[c] = inv_dydy * (vec_jp[c] - 2.0f*vec_poi[c] + vec_jn[c]);
                    d2_dz2[c] = inv_dzdz * (vec_kp[c] - 2.0f*vec_poi[c] + vec_kn[c]);

                    d2_dxdy[c] = inv_dxdy * (
                            (vec_ijp[c] + 2.0f*vec_poi[c] + vec_ijn[c]) +
                            (vec_ip[c] + vec_in[c] + vec_jp[c] + vec_jn[c])
                        );
                    d2_dxdz[c] = inv_dxdz * (
                            (vec_ikp[c] + 2.0f*vec_poi[c] + vec_ikn[c]) +
                            (vec_ip[c] + vec_in[c] + vec_kp[c] + vec_kn[c])
                        );
                    d2_dydz[c] = inv_dydz * (
                            (vec_jkp[c] + 2.0f*vec_poi[c] + vec_jkn[c]) +
                            (vec_jp[c] + vec_jn[c] + vec_kp[c] + vec_kn[c])
                        );

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
