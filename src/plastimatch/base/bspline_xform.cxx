/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "plmsys.h"

#include "bspline.h"
#include "bspline_xform.h"
#include "interpolate_macros.h"
#include "plm_math.h"
#include "plm_path.h"
#include "volume.h"
#include "volume_header.h"
#include "volume_macros.h"
#include "xpm.h"

/* BEGIN : Functions migrated from bspline.cxx */
void
bspline_interp_pix (
    float out[3], 
    const Bspline_xform* bxf, 
    plm_long p[3], 
    plm_long qidx
) {
    int i, j, k, m;
    plm_long cidx;
    float* q_lut = &bxf->q_lut[qidx*64];

    out[0] = out[1] = out[2] = 0;
    m = 0;
    for (k = 0; k < 4; k++) {
        for (j = 0; j < 4; j++) {
            for (i = 0; i < 4; i++) {
                cidx = (p[2] + k) * bxf->cdims[1] * bxf->cdims[0]
                        + (p[1] + j) * bxf->cdims[0]
                        + (p[0] + i);
                cidx = cidx * 3;
                out[0] += q_lut[m] * bxf->coeff[cidx+0];
                out[1] += q_lut[m] * bxf->coeff[cidx+1];
                out[2] += q_lut[m] * bxf->coeff[cidx+2];
                m ++;
            }
        }
    }
}

void
bspline_interp_pix_b (
    float out[3], 
    Bspline_xform* bxf, 
    plm_long pidx, 
    plm_long qidx
)
{
    int i, j, k, m;
    plm_long cidx;
    float* q_lut = &bxf->q_lut[qidx*64];
    plm_long* c_lut = &bxf->c_lut[pidx*64];

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

void
bspline_interp_pix_c (
    float out[3],
    Bspline_xform* bxf,
    plm_long pidx,
    plm_long *q
)
{
    int i,j,k,m;
    plm_long cidx;
    float A,B,C;
    plm_long* c_lut = &bxf->c_lut[pidx*64];
    float* bx_lut = &bxf->bx_lut[q[0]*4];
    float* by_lut = &bxf->by_lut[q[1]*4];
    float* bz_lut = &bxf->bz_lut[q[2]*4];

    out[0] = out[1] = out[2] = 0;
    m=0;
    for (k=0; k<4; k++) {
        C = bz_lut[k];
        for (j=0; j<4; j++) {
            B = by_lut[j] * C;
            for (i=0; i<4; i++) {
                A = bx_lut[i] * B;

                cidx = 3*c_lut[m++];
                out[0] += A * bxf->coeff[cidx+0];
                out[1] += A * bxf->coeff[cidx+1];
                out[2] += A * bxf->coeff[cidx+2];
            }
        }
    }
}

void
bspline_interpolate_vf (Volume* interp, 
    const Bspline_xform* bxf)
{
    plm_long i, j, k, v;
    plm_long p[3];
    plm_long q[3];
    float* out;
    float* img = (float*) interp->img;
    plm_long qidx;

    memset (img, 0, interp->npix*3*sizeof(float));
    for (k = 0; k < bxf->roi_dim[2]; k++) {
        p[2] = k / bxf->vox_per_rgn[2];
        q[2] = k % bxf->vox_per_rgn[2];
        for (j = 0; j < bxf->roi_dim[1]; j++) {
            p[1] = j / bxf->vox_per_rgn[1];
            q[1] = j % bxf->vox_per_rgn[1];
            for (i = 0; i < bxf->roi_dim[0]; i++) {
                p[0] = i / bxf->vox_per_rgn[0];
                q[0] = i % bxf->vox_per_rgn[0];
                qidx = volume_index (bxf->vox_per_rgn, q);
                v = (k+bxf->roi_offset[2]) * interp->dim[0] * interp->dim[1]
                    + (j+bxf->roi_offset[1]) * interp->dim[0] 
                    + (i+bxf->roi_offset[0]);
                out = &img[3*v];
                bspline_interp_pix (out, bxf, p, qidx);
            }
        }
    }
}


/* This function uses the B-Spline coefficients to transform a point.  
   The point need not lie exactly on a voxel, so we do not use the 
   lookup table. */
void
bspline_transform_point (
    float point_out[3], /* Output coordinate of point */
    Bspline_xform* bxf, /* Bspline transform coefficients */
    float point_in[3],  /* Input coordinate of point */
    int linear_interp   /* 1 = trilinear, 0 = nearest neighbors */
)
{
    plm_long d, i, j, k;
    plm_long p[3];                    /* Index of tile */
    float q[3];                  /* Fractional offset within tile */
    float q_mini[3][4];          /* "miniature" q-lut, just for this point */

    /* Default value is untransformed point */
    for (d = 0; d < 3; d++) {
        point_out[d] = point_in[d];
    }

    /* Compute tile and offset within tile */
    for (d = 0; d < 3; d++) {
        float img_ijk[3];         /* Voxel coordinate of point_in */
        img_ijk[d] = (point_in[d] - bxf->img_origin[d]) / bxf->img_spacing[d];
        p[d] = (int) floorf (
            (img_ijk[d] - bxf->roi_offset[d]) / bxf->vox_per_rgn[d]);
        /* If point lies outside of B-spline domain, return point_in */
        if (p[d] < 0 || p[d] >= bxf->rdims[d]) {
            printf ("Unwarped point, outside roi: %f %f %f\n", 
                point_out[0], point_out[1], point_out[2]);
            return;
        }
        q[d] = ((img_ijk[d] - bxf->roi_offset[d])
            - p[d] * bxf->vox_per_rgn[d]) / bxf->vox_per_rgn[d];
    }

#if defined (commentout)
    printf ("p = [%d, %d, %d], q = [%f, %f, %f]\n", 
        p[0], p[1], p[2], q[0], q[1], q[2]);
#endif

    /* Compute basis function values for this offset */
    for (d = 0; d < 3; d++) {
        float t3 = q[d]*q[d]*q[d];
        float t2 = q[d]*q[d];
        float t1 = q[d];
        q_mini[d][0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
        q_mini[d][1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
        q_mini[d][2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
        q_mini[d][3] = (1.0/6.0) * (+ 1.0 * t3);
    }

    /* Compute displacement vector and add to point_out */
#if defined (commentout)
    printf ("---\n");
#endif
    for (k = 0; k < 4; k++) {
        for (j = 0; j < 4; j++) {
            for (i = 0; i < 4; i++) {
                float ql;
                int cidx;

                cidx = (p[2] + k) * bxf->cdims[1] * bxf->cdims[0]
                    + (p[1] + j) * bxf->cdims[0]
                    + (p[0] + i);
                cidx = cidx * 3;
                ql = q_mini[0][i] * q_mini[1][j] * q_mini[2][k];

#if defined (commentout)
                printf ("(%f) + [%f] + [%f] = ", point_out[0],
                    ql, bxf->coeff[cidx+0]);
#endif

                point_out[0] += ql * bxf->coeff[cidx+0];
                point_out[1] += ql * bxf->coeff[cidx+1];
                point_out[2] += ql * bxf->coeff[cidx+2];

#if defined (commentout)
                printf (" = (%f)\n", point_out[0]);
#endif
            }
        }
    }
}

/* END : Functions migrated from bspline.cxx */
/*---------------------*/

static float
bspline_basis_eval (
    int t_idx, 
    int vox_idx, 
    int vox_per_rgn)
{
                                
    float i = (float)vox_idx / vox_per_rgn;

    switch(t_idx) {
    case 0:
        return (1.0/6.0) * (- 1.0 * i*i*i + 3.0 * i*i - 3.0 * i + 1.0);
        break;
    case 1:
        return (1.0/6.0) * (+ 3.0 * i*i*i - 6.0 * i*i           + 4.0);
        break;
    case 2:
        return (1.0/6.0) * (- 3.0 * i*i*i + 3.0 * i*i + 3.0 * i + 1.0);
        break;
    case 3:
        return (1.0/6.0) * (+ 1.0 * i*i*i);
        break;
    default:
        return 0.0;
        break;
    }
}

void
bspline_xform_set_default (Bspline_xform* bxf)
{
    int d;

    memset (bxf, 0, sizeof (Bspline_xform));

    for (d = 0; d < 3; d++) {
        bxf->img_origin[d] = 0.0f;
        bxf->img_spacing[d] = 1.0f;
        bxf->img_dim[d] = 0;
        bxf->roi_offset[d] = 0;
        bxf->roi_dim[d] = 0;
        bxf->vox_per_rgn[d] = 30;
        bxf->grid_spac[d] = 30.0f;
    }
}

void
bspline_xform_save (Bspline_xform* bxf, const char* filename)
{
    FILE* fp;

    make_directory_recursive (filename);
    fp = fopen (filename, "wb");
    if (!fp) return;

    fprintf (fp, "MGH_GPUIT_BSP <experimental>\n");
    fprintf (fp, "img_origin = %f %f %f\n", 
        bxf->img_origin[0], bxf->img_origin[1], bxf->img_origin[2]);
    fprintf (fp, "img_spacing = %f %f %f\n", 
        bxf->img_spacing[0], bxf->img_spacing[1], bxf->img_spacing[2]);
    fprintf (fp, "img_dim = %u %u %u\n", 
        (unsigned int) bxf->img_dim[0], (unsigned int) bxf->img_dim[1], 
        (unsigned int) bxf->img_dim[2]);
    fprintf (fp, "roi_offset = %d %d %d\n", 
        (unsigned int) bxf->roi_offset[0], (unsigned int) bxf->roi_offset[1], 
        (unsigned int) bxf->roi_offset[2]);
    fprintf (fp, "roi_dim = %d %d %d\n", 
        (unsigned int) bxf->roi_dim[0], (unsigned int) bxf->roi_dim[1], 
        (unsigned int) bxf->roi_dim[2]);
    fprintf (fp, "vox_per_rgn = %d %d %d\n", 
        (unsigned int) bxf->vox_per_rgn[0], 
        (unsigned int) bxf->vox_per_rgn[1], 
        (unsigned int) bxf->vox_per_rgn[2]);
    fprintf (fp, "direction_cosines = %f %f %f %f %f %f %f %f %f\n", 
        (bxf->dc).m_direction_cosines[0], 
        (bxf->dc).m_direction_cosines[1], 
        (bxf->dc).m_direction_cosines[2], 
        (bxf->dc).m_direction_cosines[3], 
        (bxf->dc).m_direction_cosines[4], 
        (bxf->dc).m_direction_cosines[5], 
        (bxf->dc).m_direction_cosines[6], 
        (bxf->dc).m_direction_cosines[7], 
        (bxf->dc).m_direction_cosines[8]);
    /* No need to save grid_spac */

#if defined (commentout)
    {
        /* This dumps in native, interleaved format */
        for (i = 0; i < bxf->num_coeff; i++) {
            fprintf (fp, "%6.3f\n", bxf->coeff[i]);
        }
    }
#endif

    /* This dumps in itk-like planar format */
    {
        int i, j;
        for (i = 0; i < 3; i++) {
            for (j = 0; j < bxf->num_coeff / 3; j++) {
                //fprintf (fp, "%6.3f\n", bxf->coeff[j*3 + i]);
                fprintf (fp, "%.20f\n", bxf->coeff[j*3 + i]);
            }
        }
    }           

    fclose (fp);
}

Bspline_xform* 
bspline_xform_load (const char* filename)
{
    Bspline_xform* bxf;
    char buf[1024];
    FILE* fp;
    int rc;
    float img_origin[3];         /* Image origin (in mm) */
    float img_spacing[3];        /* Image spacing (in mm) */
    unsigned int a, b, c;        /* For fscanf */
    plm_long img_dim[3];           /* Image size (in vox) */
    plm_long roi_offset[3];      /* Position of first vox in ROI (in vox) */
    plm_long roi_dim[3];                 /* Dimension of ROI (in vox) */
    plm_long vox_per_rgn[3];     /* Knot spacing (in vox) */
    float dc[9];                 /* Direction cosines */

    fp = fopen (filename, "r");
    if (!fp) return 0;

    /* Initialize parms */
    bxf = (Bspline_xform*) malloc (sizeof(Bspline_xform));
    bspline_xform_set_default (bxf);

    /* Skip first line */
    fgets (buf, 1024, fp);

    /* Read header */
    rc = fscanf (fp, "img_origin = %f %f %f\n", 
        &img_origin[0], &img_origin[1], &img_origin[2]);
    if (rc != 3) {
        logfile_printf ("Error parsing input xform (img_origin): %s\n", filename);
        goto free_exit;
    }
    rc = fscanf (fp, "img_spacing = %f %f %f\n", 
        &img_spacing[0], &img_spacing[1], &img_spacing[2]);
    if (rc != 3) {
        logfile_printf ("Error parsing input xform (img_spacing): %s\n", filename);
        goto free_exit;
    }
    rc = fscanf (fp, "img_dim = %d %d %d\n", &a, &b, &c);
    if (rc != 3) {
        logfile_printf ("Error parsing input xform (img_dim): %s\n", filename);
        goto free_exit;
    }
    img_dim[0] = a;
    img_dim[1] = b;
    img_dim[2] = c;

    rc = fscanf (fp, "roi_offset = %d %d %d\n", &a, &b, &c);
    if (rc != 3) {
        logfile_printf ("Error parsing input xform (roi_offset): %s\n", filename);
        goto free_exit;
    }
    roi_offset[0] = a;
    roi_offset[1] = b;
    roi_offset[2] = c;

    rc = fscanf (fp, "roi_dim = %d %d %d\n", &a, &b, &c);
    if (rc != 3) {
        logfile_printf ("Error parsing input xform (roi_dim): %s\n", filename);
        goto free_exit;
    }
    roi_dim[0] = a;
    roi_dim[1] = b;
    roi_dim[2] = c;

    rc = fscanf (fp, "vox_per_rgn = %d %d %d\n", &a, &b, &c);
    if (rc != 3) {
        logfile_printf ("Error parsing input xform (vox_per_rgn): %s\n", filename);
        goto free_exit;
    }
    vox_per_rgn[0] = a;
    vox_per_rgn[1] = b;
    vox_per_rgn[2] = c;

    /* JAS 2012.03.29 : check for direction cosines
     * we must be careful because older plastimatch xforms do not have this */
    rc = fscanf (fp, "direction_cosines = %f %f %f %f %f %f %f %f %f\n",
            &dc[0], &dc[1], &dc[2], &dc[3], &dc[4],
            &dc[5], &dc[6], &dc[7], &dc[8]);
    if (rc != 9) {
        dc[0] = 1.; dc[3] = 0.; dc[6] = 0.;
        dc[1] = 0.; dc[4] = 1.; dc[7] = 0.;
        dc[2] = 0.; dc[5] = 0.; dc[8] = 1.;
    }
    

    /* Allocate memory and build LUTs */
    bspline_xform_initialize (bxf, img_origin, img_spacing, img_dim,
        roi_offset, roi_dim, vox_per_rgn, dc);

    /* This loads from itk-like planar format */
    {
        int i, j;
        for (i = 0; i < 3; i++) {
            for (j = 0; j < bxf->num_coeff / 3; j++) {
                rc = fscanf (fp, "%f\n", &bxf->coeff[j*3 + i]);
                if (rc != 1) {
                    logfile_printf ("Error parsing input xform (idx = %d,%d): %s\n", i, j, filename);
                    bspline_xform_free (bxf);
                    goto free_exit;
                }
            }
        }
    }

    fclose (fp);
    return bxf;

  free_exit:
    fclose (fp);
    free (bxf);
    return 0;
}


/* -----------------------------------------------------------------------
   Debugging routines
   ----------------------------------------------------------------------- */
void
bspline_xform_dump_coeff (Bspline_xform* bxf, const char* fn)
{
    int i;
    FILE* fp = fopen (fn,"wb");
    for (i = 0; i < bxf->num_coeff; i++) {
        fprintf (fp, "%20.20f\n", bxf->coeff[i]);
    }
    fclose (fp);
}

void
bspline_xform_dump_luts (Bspline_xform* bxf)
{
    plm_long i, j, k, p;
    int tx, ty, tz;
    FILE* fp = fopen ("qlut.txt","wb");

    /* Dump q_lut */
    for (k = 0, p = 0; k < bxf->vox_per_rgn[2]; k++) {
        for (j = 0; j < bxf->vox_per_rgn[1]; j++) {
            for (i = 0; i < bxf->vox_per_rgn[0]; i++) {
                fprintf (fp, "%3d %3d %3d\n", 
                    (unsigned int) k, (unsigned int) j, (unsigned int) i);
                for (tz = 0; tz < 4; tz++) {
                    for (ty = 0; ty < 4; ty++) {
                        for (tx = 0; tx < 4; tx++) {
                            fprintf (fp, " %f", bxf->q_lut[p++]);
                        }
                    }
                }
                fprintf (fp, "\n");
            }
        }
    }
    fclose (fp);

    /* Test q_lut */
#if defined (commentout)
    printf ("Testing q_lut\n");
    for (j = 0; j < bxf->vox_per_rgn[2] 
                 * bxf->vox_per_rgn[1] 
                 * bxf->vox_per_rgn[0]; j++) {
        float sum = 0.0;
        for (i = j*64; i < (j+1)*64; i++) {
            sum += bxf->q_lut[i];
        }
        if (fabs(sum-1.0) > 1.e-7) {
            printf ("%g ", fabs(sum-1.0));
        }
    }
    printf ("\n");
#endif

    fp = fopen ("clut.txt","wb");
    p = 0;
    for (k = 0; k < bxf->rdims[2]; k++) {
        for (j = 0; j < bxf->rdims[1]; j++) {
            for (i = 0; i < bxf->rdims[0]; i++) {
                                
                fprintf (fp, "%3u %3u %3u\n", 
                    (unsigned int) k, (unsigned int) j, (unsigned int) i);
                
                for (tz = 0; tz < 4; tz++) {
                    for (ty = 0; ty < 4; ty++) {
                        for (tx = 0; tx < 4; tx++) {
                            fprintf (fp, " %u", (unsigned int) bxf->c_lut[p++]);
                        }
                    }
                }
                fprintf (fp, "\n");
            }
        }
    }
    fclose (fp);
}

void
bspline_xform_set_coefficients (Bspline_xform* bxf, float val)
{
    int i;

    for (i = 0; i < bxf->num_coeff; i++) {
        bxf->coeff[i] = val;
    }
}

void
bspline_xform_initialize 
(
    Bspline_xform* bxf,           /* Output: bxf is initialized */
    float img_origin[3],          /* Image origin (in mm) */
    float img_spacing[3],         /* Image spacing (in mm) */
    plm_long img_dim[3],          /* Image size (in vox) */
    plm_long roi_offset[3],       /* Position of first vox in ROI (in vox) */
    plm_long roi_dim[3],          /* Dimension of ROI (in vox) */
    plm_long vox_per_rgn[3],      /* Knot spacing (in vox) */
    float direction_cosines[9]    /* Direction cosines */
)
{
    plm_long d;
    plm_long i, j, k, p;
    plm_long tx, ty, tz;
    float *A, *B, *C;

    logfile_printf ("bspline_xform_initialize\n");
    for (d = 0; d < 3; d++) {
        /* copy input parameters over */
        bxf->img_origin[d] = img_origin[d];
        bxf->img_spacing[d] = img_spacing[d];
        bxf->img_dim[d] = img_dim[d];
        bxf->roi_offset[d] = roi_offset[d];
        bxf->roi_dim[d] = roi_dim[d];
        bxf->vox_per_rgn[d] = vox_per_rgn[d];
        bxf->dc.set (direction_cosines);

        /* grid spacing is in mm */
        bxf->grid_spac[d] = bxf->vox_per_rgn[d] * fabs (bxf->img_spacing[d]);

        /* rdims is the number of regions */
        bxf->rdims[d] = 1 + (bxf->roi_dim[d] - 1) / bxf->vox_per_rgn[d];

        /* cdims is the number of control points */
        bxf->cdims[d] = 3 + bxf->rdims[d];
    }

    /* total number of control points & coefficients */
    bxf->num_knots = bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2];
    bxf->num_coeff = bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2] * 3;

    /* Allocate coefficients */
    bxf->coeff = (float*) malloc (sizeof(float) * bxf->num_coeff);
    memset (bxf->coeff, 0, sizeof(float) * bxf->num_coeff);

    /* Create q_lut */
    bxf->q_lut = (float*) malloc (sizeof(float) 
        * bxf->vox_per_rgn[0] 
        * bxf->vox_per_rgn[1] 
        * bxf->vox_per_rgn[2] 
        * 64);
    if (!bxf->q_lut) {
        print_and_exit ("Error allocating memory for q_lut\n");
    }

    A = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[0] * 4);
    B = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[1] * 4);
    C = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[2] * 4);

    for (i = 0; i < bxf->vox_per_rgn[0]; i++) {
        float ii = ((float) i) / bxf->vox_per_rgn[0];
        float t3 = ii*ii*ii;
        float t2 = ii*ii;
        float t1 = ii;
        A[i*4+0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
        A[i*4+1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
        A[i*4+2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
        A[i*4+3] = (1.0/6.0) * (+ 1.0 * t3);
    }

    for (j = 0; j < bxf->vox_per_rgn[1]; j++) {
        float jj = ((float) j) / bxf->vox_per_rgn[1];
        float t3 = jj*jj*jj;
        float t2 = jj*jj;
        float t1 = jj;
        B[j*4+0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
        B[j*4+1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
        B[j*4+2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
        B[j*4+3] = (1.0/6.0) * (+ 1.0 * t3);
    }

    for (k = 0; k < bxf->vox_per_rgn[2]; k++) {
        float kk = ((float) k) / bxf->vox_per_rgn[2];
        float t3 = kk*kk*kk;
        float t2 = kk*kk;
        float t1 = kk;
        C[k*4+0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
        C[k*4+1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
        C[k*4+2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
        C[k*4+3] = (1.0/6.0) * (+ 1.0 * t3);
    }

    p = 0;
    for (k = 0; k < bxf->vox_per_rgn[2]; k++) {
        for (j = 0; j < bxf->vox_per_rgn[1]; j++) {
            for (i = 0; i < bxf->vox_per_rgn[0]; i++) {
                for (tz = 0; tz < 4; tz++) {
                    for (ty = 0; ty < 4; ty++) {
                        for (tx = 0; tx < 4; tx++) {
                            bxf->q_lut[p++] = A[i*4+tx] * B[j*4+ty] * C[k*4+tz];
                        }
                    }
                }
            }
        }
    }
    free (C);
    free (B);
    free (A);
        
    /* Create c_lut */
    bxf->c_lut = (plm_long*) malloc (sizeof(plm_long) 
        * bxf->rdims[0] 
        * bxf->rdims[1] 
        * bxf->rdims[2] 
        * 64);
    p = 0;
    for (k = 0; k < bxf->rdims[2]; k++) {
        for (j = 0; j < bxf->rdims[1]; j++) {
            for (i = 0; i < bxf->rdims[0]; i++) {
                for (tz = 0; tz < 4; tz++) {
                    for (ty = 0; ty < 4; ty++) {
                        for (tx = 0; tx < 4; tx++) {
                            bxf->c_lut[p++] = 
                                + (k + tz) * bxf->cdims[0] * bxf->cdims[1]
                                + (j + ty) * bxf->cdims[0] 
                                + (i + tx);
                        }
                    }
                }
            }
        }
    }

    /* Create b_luts */
    bxf->bx_lut = (float*)malloc(4*bxf->vox_per_rgn[0]*sizeof(float));
    bxf->by_lut = (float*)malloc(4*bxf->vox_per_rgn[1]*sizeof(float));
    bxf->bz_lut = (float*)malloc(4*bxf->vox_per_rgn[2]*sizeof(float));

    for (int j=0; j<4; j++) {
        for (int i=0; i<bxf->vox_per_rgn[0]; i++) {
            bxf->bx_lut[i*4+j] = bspline_basis_eval (j, i, bxf->vox_per_rgn[0]);
        }
        for (int i=0; i<bxf->vox_per_rgn[1]; i++) {
            bxf->by_lut[i*4+j] = bspline_basis_eval (j, i, bxf->vox_per_rgn[1]);
        }
        for (int i=0; i<bxf->vox_per_rgn[2]; i++) {
            bxf->bz_lut[i*4+j] = bspline_basis_eval (j, i, bxf->vox_per_rgn[2]);
        }
    }

    //dump_luts (bxf);

    logfile_printf ("rdims = (%d,%d,%d)\n", 
        bxf->rdims[0], bxf->rdims[1], bxf->rdims[2]);
    logfile_printf ("vox_per_rgn = (%d,%d,%d)\n", 
        bxf->vox_per_rgn[0], bxf->vox_per_rgn[1], bxf->vox_per_rgn[2]);
    logfile_printf ("cdims = (%d %d %d)\n", 
        bxf->cdims[0], bxf->cdims[1], bxf->cdims[2]);
}

/* -----------------------------------------------------------------------
   This extends the bspline grid.  Note, that the new roi_offset 
    in the bxf will not be the same as the one requested, because 
    bxf routines implicitly require that the first voxel of the 
    ROI matches the position of the control point. 
   ----------------------------------------------------------------------- */
/* GCS -- Is there an implicit assumption that the roi_origin > 0? */
void
bspline_xform_extend (
    Bspline_xform* bxf,      /* Output: bxf is initialized */
    int new_roi_offset[3],   /* Position of first vox in ROI (in vox) */
    int new_roi_dim[3]       /* Dimension of ROI (in vox) */
)
{
    int d;
    int roi_offset_diff[3];
    int roi_corner_diff[3];
    int eb[3];  /* # of control points to "extend before" existing grid */
    int ea[3];  /* # of control points to "extend after" existing grid */
    int extend_needed = 0;
    int new_rdims[3];
    int new_cdims[3];
    plm_long new_num_knots;
    plm_long new_num_coeff;
    float* new_coeff;
    plm_long old_idx;
    plm_long i, j, k;

    for (d = 0; d < 3; d++) {
        roi_offset_diff[d] = new_roi_offset[d] - bxf->roi_offset[d];
        roi_corner_diff[d] = (new_roi_offset[d] + new_roi_dim[d]) 
            - (bxf->roi_offset[d] + bxf->roi_offset[d]);

        if (roi_offset_diff[d] < 0) {
            eb[d] = (bxf->vox_per_rgn[d] - roi_offset_diff[d] - 1) 
                / bxf->vox_per_rgn[d];
            extend_needed = 1;
        } else {
            eb[d] = 0;
        }
        if (roi_corner_diff[d] > 0) {
            ea[d] = (bxf->vox_per_rgn[d] + roi_corner_diff[d] - 1) 
                / bxf->vox_per_rgn[d];
            extend_needed = 1;
        } else {
            ea[d] = 0;
        }
    }

    if (extend_needed) {
        /* Allocate new memory */
        for (d = 0; d < 3; d++) {
            new_rdims[d] = bxf->rdims[d] + ea[d] + eb[d];
            new_cdims[d] = bxf->cdims[d] + ea[d] + eb[d];
        }
        new_num_knots = bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2];
        new_num_coeff = bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2] * 3;
        new_coeff = (float*) malloc (sizeof(float) * new_num_coeff);
        memset (new_coeff, 0, sizeof(float) * new_num_coeff);

        /* Copy coefficients to new memory */
        for (old_idx = 0, k = 0; k < bxf->cdims[2]; k++) {
            for (j = 0; j < bxf->cdims[1]; j++) {
                for (i = 0; i < bxf->cdims[0]; i++) {
                    plm_long new_idx = 3 * (((((k + eb[2]) * new_cdims[1]) + (j + eb[1])) * new_cdims[0]) + (i + eb[0]));
                    for (d = 0; d < 3; d++, old_idx++, new_idx++) {
                        new_coeff[new_idx] = bxf->coeff[old_idx];
                    }
                }
            }
        }

        /* Free old memory */
        free (bxf->coeff);

        /* Copy over new data into bxf */
        for (d = 0; d < 3; d++) {
            bxf->rdims[d] = new_rdims[d];
            bxf->cdims[d] = new_cdims[d];
        }
        bxf->num_knots = new_num_knots;
        bxf->num_coeff = new_num_coeff;
        bxf->coeff = new_coeff;

        /* Special consideration to ROI */
        for (d = 0; d < 3; d++) {
            bxf->roi_offset[d] = bxf->roi_offset[d] - eb[d] * bxf->vox_per_rgn[d];
            bxf->roi_dim[d] = new_roi_dim[d] + (new_roi_offset[d] - bxf->roi_offset[d]);
        }
    }
}

void
bspline_xform_free (Bspline_xform* bxf)
{
    free (bxf->coeff);
    free (bxf->q_lut);
    free (bxf->c_lut);
    free (bxf->bx_lut);
    free (bxf->by_lut);
    free (bxf->bz_lut);
}

/* Set volume header from B-spline Xform */
void 
Bspline_xform::get_volume_header (Volume_header *vh)
{
#if 0
    vh->set_dim (this->img_dim);
    vh->set_origin (this->img_origin);
    vh->set_spacing (this->img_spacing);
#endif

    vh->set (this->img_dim, this->img_origin, this->img_spacing, (this->dc).m_direction_cosines);
}
