/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "bspline.h"
#include "interpolate.h"
#include "logfile.h"
#include "math_util.h"
#include "mha_io.h"
#include "plm_path.h"
#include "plm_timer.h"
#include "print_and_exit.h"
#include "volume.h"
#include "volume_macros.h"
#include "xpm.h"

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
write_bxf (const char* filename, Bspline_xform* bxf)
{
    FILE* fp;
	
    fp = fopen (filename, "wb");
    if (!fp) return;

    fprintf (fp, "MGH_GPUIT_BSP <experimental>\n");
    fprintf (fp, "img_origin = %f %f %f\n", 
	bxf->img_origin[0], bxf->img_origin[1], bxf->img_origin[2]);
    fprintf (fp, "img_spacing = %f %f %f\n", 
	bxf->img_spacing[0], bxf->img_spacing[1], bxf->img_spacing[2]);
    fprintf (fp, "img_dim = %d %d %d\n", 
	bxf->img_dim[0], bxf->img_dim[1], bxf->img_dim[2]);
    fprintf (fp, "roi_offset = %d %d %d\n", 
	bxf->roi_offset[0], bxf->roi_offset[1], bxf->roi_offset[2]);
    fprintf (fp, "roi_dim = %d %d %d\n", 
	bxf->roi_dim[0], bxf->roi_dim[1], bxf->roi_dim[2]);
    fprintf (fp, "vox_per_rgn = %d %d %d\n", 
	bxf->vox_per_rgn[0], bxf->vox_per_rgn[1], bxf->vox_per_rgn[2]);
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
		fprintf (fp, "%f\n", bxf->coeff[j*3 + i]);
	    }
	}
    }		

    fclose (fp);
}

Bspline_xform* 
read_bxf (char* filename)
{
    Bspline_xform* bxf;
    char buf[1024];
    FILE* fp;
    int rc;
    float img_origin[3];         /* Image origin (in mm) */
    float img_spacing[3];        /* Image spacing (in mm) */
    int img_dim[3];              /* Image size (in vox) */
    int roi_offset[3];		 /* Position of first vox in ROI (in vox) */
    int roi_dim[3];		 /* Dimension of ROI (in vox) */
    int vox_per_rgn[3];		 /* Knot spacing (in vox) */

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
    rc = fscanf (fp, "img_dim = %d %d %d\n", 
	&img_dim[0], &img_dim[1], &img_dim[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (img_dim): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "roi_offset = %d %d %d\n", 
	&roi_offset[0], &roi_offset[1], &roi_offset[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (roi_offset): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "roi_dim = %d %d %d\n", 
	&roi_dim[0], &roi_dim[1], &roi_dim[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (roi_dim): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "vox_per_rgn = %d %d %d\n", 
	&vox_per_rgn[0], &vox_per_rgn[1], &vox_per_rgn[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (vox_per_rgn): %s\n", filename);
	goto free_exit;
    }

    /* Allocate memory and build LUTs */
    bspline_xform_initialize (bxf, img_origin, img_spacing, img_dim,
		roi_offset, roi_dim, vox_per_rgn);

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
dump_coeff (Bspline_xform* bxf, char* fn)
{
    int i;
    FILE* fp = fopen (fn,"wb");
    for (i = 0; i < bxf->num_coeff; i++) {
	fprintf (fp, "%f\n", bxf->coeff[i]);
    }
    fclose (fp);
}

void
dump_luts (Bspline_xform* bxf)
{
    int i, j, k, p;
    int tx, ty, tz;
    FILE* fp = fopen ("qlut.txt","wb");

    /* Dump q_lut */
    for (k = 0, p = 0; k < bxf->vox_per_rgn[2]; k++) {
	for (j = 0; j < bxf->vox_per_rgn[1]; j++) {
	    for (i = 0; i < bxf->vox_per_rgn[0]; i++) {
		fprintf (fp, "%3d %3d %3d\n", k, j, i);
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
				
		fprintf (fp, "%3d %3d %3d\n", k, j, i);
		
		for (tz = 0; tz < 4; tz++) {
		    for (ty = 0; ty < 4; ty++) {
			for (tx = 0; tx < 4; tx++) {
			    fprintf (fp, " %d", bxf->c_lut[p++]);
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
bspline_set_coefficients (Bspline_xform* bxf, float val)
{
    int i;

    for (i = 0; i < bxf->num_coeff; i++) {
	bxf->coeff[i] = val;
    }
}

void
bspline_xform_initialize 
(
    Bspline_xform* bxf,         /* Output: bxf is initialized */
    float img_origin[3],        /* Image origin (in mm) */
    float img_spacing[3],       /* Image spacing (in mm) */
    int img_dim[3],             /* Image size (in vox) */
    int roi_offset[3],          /* Position of first vox in ROI (in vox) */
    int roi_dim[3],             /* Dimension of ROI (in vox) */
    int vox_per_rgn[3])         /* Knot spacing (in vox) */
{
    int d;
    int i, j, k, p;
    int tx, ty, tz;
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

	/* grid spacing is in mm */
	bxf->grid_spac[d] = bxf->vox_per_rgn[d] * bxf->img_spacing[d];

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
    bxf->c_lut = (int*) malloc (sizeof(int) 
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

    //dump_luts (bxf);

    logfile_printf ("rdims = (%d,%d,%d)\n", 
	bxf->rdims[0], bxf->rdims[1], bxf->rdims[2]);
    logfile_printf ("vox_per_rgn = (%d,%d,%d)\n", 
	bxf->vox_per_rgn[0], bxf->vox_per_rgn[1], bxf->vox_per_rgn[2]);
    logfile_printf ("cdims = (%d %d %d)\n", 
	bxf->cdims[0], bxf->cdims[1], bxf->cdims[2]);
}

void bspline_xform_create_qlut_grad 
(
    Bspline_xform* bxf,         /* Output: bxf with new LUTs */
    float img_spacing[3],       /* Image spacing (in mm) */
    int vox_per_rgn[3])         /* Knot spacing (in vox) */
{
    int i, j, k, p;
    int tx, ty, tz;
    float *A, *B, *C;
    float *Ax, *By, *Cz, *Axx, *Byy, *Czz;
    int q_lut_size;

    q_lut_size = sizeof(float) * bxf->vox_per_rgn[0] 
	* bxf->vox_per_rgn[1] 
	* bxf->vox_per_rgn[2] 
	* 64;
    logfile_printf("Creating gradient multiplier LUTs, %d bytes each\n", q_lut_size);

    bxf->q_dxdyz_lut = (float*) malloc ( q_lut_size );
    if (!bxf->q_dxdyz_lut) print_and_exit ("Error allocating memory for q_grad_lut\n");
	
    bxf->q_xdydz_lut = (float*) malloc ( q_lut_size );
    if (!bxf->q_xdydz_lut) print_and_exit ("Error allocating memory for q_grad_lut\n");

    bxf->q_dxydz_lut = (float*) malloc ( q_lut_size );
    if (!bxf->q_dxydz_lut) print_and_exit ("Error allocating memory for q_grad_lut\n");
	
    bxf->q_d2xyz_lut = (float*) malloc ( q_lut_size );
    if (!bxf->q_d2xyz_lut) print_and_exit ("Error allocating memory for q_grad_lut\n");

    bxf->q_xd2yz_lut = (float*) malloc ( q_lut_size );
    if (!bxf->q_xd2yz_lut) print_and_exit ("Error allocating memory for q_grad_lut\n");

    bxf->q_xyd2z_lut = (float*) malloc ( q_lut_size );
    if (!bxf->q_xyd2z_lut) print_and_exit ("Error allocating memory for q_grad_lut\n");

    A = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[0] * 4);
    B = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[1] * 4);
    C = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[2] * 4);

    Ax = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[0] * 4);
    By = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[1] * 4);
    Cz = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[2] * 4);

    Axx = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[0] * 4);
    Byy = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[1] * 4);
    Czz = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[2] * 4);

    for (i = 0; i < bxf->vox_per_rgn[0]; i++) {
	float ii = ((float) i) / bxf->vox_per_rgn[0];
	float t3 = ii*ii*ii;
	float t2 = ii*ii;
	float t1 = ii;
	A[i*4+0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	A[i*4+1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	A[i*4+2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	A[i*4+3] = (1.0/6.0) * (+ 1.0 * t3);

	Ax[i*4+0] =(1.0/6.0) * (- 3.0 * t2 + 6.0 * t1 - 3.0           );
	Ax[i*4+1] =(1.0/6.0) * (+ 9.0 * t2 - 12.0* t1                 );
	Ax[i*4+2] =(1.0/6.0) * (- 9.0 * t2 + 6.0 * t1 + 3.0           );
	Ax[i*4+3] =(1.0/6.0) * (+ 3.0 * t2);

	Axx[i*4+0]=(1.0/6.0) * (- 6.0 * t1 + 6.0                     );
	Axx[i*4+1]=(1.0/6.0) * (+18.0 * t1 - 12.0                    );
	Axx[i*4+2]=(1.0/6.0) * (-18.0 * t1 + 6.0                     );
	Axx[i*4+3]=(1.0/6.0) * (+ 6.0 * t1);
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

	By[j*4+0] =(1.0/6.0) * (- 3.0 * t2 + 6.0 * t1 - 3.0           );
	By[j*4+1] =(1.0/6.0) * (+ 9.0 * t2 - 12.0* t1                 );
	By[j*4+2] =(1.0/6.0) * (- 9.0 * t2 + 6.0 * t1 + 3.0           );
	By[j*4+3] =(1.0/6.0) * (+ 3.0 * t2);

	Byy[j*4+0]=(1.0/6.0) * (- 6.0 * t1 + 6.0                     );
	Byy[j*4+1]=(1.0/6.0) * (+18.0 * t1 - 12.0                    );
	Byy[j*4+2]=(1.0/6.0) * (-18.0 * t1 + 6.0                     );
	Byy[j*4+3]=(1.0/6.0) * (+ 6.0 * t1);
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

	Cz[k*4+0] =(1.0/6.0) * (- 3.0 * t2 + 6.0 * t1 - 3.0           );
	Cz[k*4+1] =(1.0/6.0) * (+ 9.0 * t2 - 12.0* t1                 );
	Cz[k*4+2] =(1.0/6.0) * (- 9.0 * t2 + 6.0 * t1 + 3.0           );
	Cz[k*4+3] =(1.0/6.0) * (+ 3.0 * t2);

	Czz[k*4+0]=(1.0/6.0) * (- 6.0 * t1 + 6.0                     );
	Czz[k*4+1]=(1.0/6.0) * (+18.0 * t1 - 12.0                    );
	Czz[k*4+2]=(1.0/6.0) * (-18.0 * t1 + 6.0                     );
	Czz[k*4+3]=(1.0/6.0) * (+ 6.0 * t1);
    }

    p = 0;
    for (k = 0; k < bxf->vox_per_rgn[2]; k++) {
	for (j = 0; j < bxf->vox_per_rgn[1]; j++) {
	    for (i = 0; i < bxf->vox_per_rgn[0]; i++) {
		for (tz = 0; tz < 4; tz++) {
		    for (ty = 0; ty < 4; ty++) {
			for (tx = 0; tx < 4; tx++) {
				
			    bxf->q_dxdyz_lut[p] = Ax[i*4+tx] * By[j*4+ty] * C[k*4+tz];
			    bxf->q_xdydz_lut[p] = A[i*4+tx] * By[j*4+ty] * Cz[k*4+tz];
			    bxf->q_dxydz_lut[p] = Ax[i*4+tx] * B[j*4+ty] * Cz[k*4+tz];

			    bxf->q_d2xyz_lut[p] = Axx[i*4+tx] * B[j*4+ty] * C[k*4+tz];
			    bxf->q_xd2yz_lut[p] = A[i*4+tx] * Byy[j*4+ty] * C[k*4+tz];
			    bxf->q_xyd2z_lut[p] = A[i*4+tx] * B[j*4+ty] * Czz[k*4+tz];

			    p++;
			}
		    }
		}
	    }
	}
    }
    free (C);
    free (B);
    free (A);
    free (Ax); free(By); free(Cz); free(Axx); free(Byy); free(Czz);
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
    Bspline_xform* bxf,	     /* Output: bxf is initialized */
    int new_roi_offset[3],   /* Position of first vox in ROI (in vox) */
    int new_roi_dim[3]	     /* Dimension of ROI (in vox) */
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
    int new_num_knots;
    int new_num_coeff;
    float* new_coeff;
    int old_idx;
    int i, j, k;

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
		    int new_idx = 3 * (((((k + eb[2]) * new_cdims[1]) + (j + eb[1])) * new_cdims[0]) + (i + eb[0]));
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
}

void
bspline_xform_free_qlut_grad (Bspline_xform* bxf)
{
    free (bxf->q_dxdyz_lut);
    free (bxf->q_dxydz_lut);
    free (bxf->q_xdydz_lut);
    free (bxf->q_d2xyz_lut);
    free (bxf->q_xd2yz_lut);
    free (bxf->q_xyd2z_lut);
}


