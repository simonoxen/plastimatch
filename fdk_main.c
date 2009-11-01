/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mathutil.h"
#include "fdk.h"
#include "readmha.h"
#include "fdk_opts.h"
#include "fdk_utils.h"

/* get_pixel_value_c seems to be no faster than get_pixel_value_b, 
   despite having two fewer compares. */
inline float
get_pixel_value_c (CB_Image* cbi, double r, double c)
{
    int rr, cc;

    r += 0.5;
    if (r < 0) return 0.0;
    rr = (int) r;
    if (rr >= cbi->dim[1]) return 0.0;

    c += 0.5;
    if (c < 0) return 0.0;
    cc = (int) c;
    if (cc >= cbi->dim[0]) return 0.0;

    return cbi->img[rr*cbi->dim[0] + cc];
}

inline float
get_pixel_value_b (CB_Image* cbi, double r, double c)
{
    int rr, cc;

    rr = ROUND_INT(r);
    if (rr < 0 || rr >= cbi->dim[1]) return 0.0;
    cc = ROUND_INT(c);
    if (cc < 0 || cc >= cbi->dim[0]) return 0.0;
    return cbi->img[rr*cbi->dim[0] + cc];
}

#if defined (commentout)
inline float
get_pixel_value_a (CB_Image* cbi, double r, double c)
{
    int rr, cc;

    rr = round_int (r);
    if (rr < 0 || rr >= cbi->dim[1]) return 0.0;
    cc = round_int (c);
    if (cc < 0 || cc >= cbi->dim[0]) return 0.0;
    return cbi->img[rr*cbi->dim[0] + cc];
}
#endif

/* This version folds ic & wip into zip, as well as using faster 
   nearest neighbor macro. */
void
project_volume_onto_image_c (Volume* vol, CB_Image* cbi, float scale)
{
    int i, j, k, p;
    float* img = (float*) vol->img;
    double *xip, *yip, *zip;
    double acc2[3],acc3[3];
    double dw;
    double sad_sid_2;

    /* Rescale image (destructive rescaling) */
    sad_sid_2 = (cbi->sad * cbi->sad) / (cbi->sid * cbi->sid);
    for (i = 0; i < cbi->dim[0]*cbi->dim[1]; i++) {
	cbi->img[i] *= sad_sid_2;	// Speedup trick re: Kachelsreiss
	cbi->img[i] *= scale;		// User scaling
    }

    xip = (double*) malloc (3*vol->dim[0]*sizeof(double));
    yip = (double*) malloc (3*vol->dim[1]*sizeof(double));
    zip = (double*) malloc (3*vol->dim[2]*sizeof(double));

    /* Precompute partial projections here */
    for (i = 0; i < vol->dim[0]; i++) {
	double x = (double) (vol->offset[0] + i * vol->pix_spacing[0]);
	xip[i*3+0] = x * (cbi->matrix[0] + cbi->ic[0] * cbi->matrix[8]);
	xip[i*3+1] = x * (cbi->matrix[4] + cbi->ic[1] * cbi->matrix[8]);
	xip[i*3+2] = x * cbi->matrix[8];
    }
    for (j = 0; j < vol->dim[1]; j++) {
	double y = (double) (vol->offset[1] + j * vol->pix_spacing[1]);
	yip[j*3+0] = y * (cbi->matrix[1] + cbi->ic[0] * cbi->matrix[9]);
	yip[j*3+1] = y * (cbi->matrix[5] + cbi->ic[1] * cbi->matrix[9]);
	yip[j*3+2] = y * cbi->matrix[9];
    }
    for (k = 0; k < vol->dim[2]; k++) {
	double z = (double) (vol->offset[2] + k * vol->pix_spacing[2]);
	zip[k*3+0] = z * (cbi->matrix[2] + cbi->ic[0] * cbi->matrix[10]) 
		+ cbi->ic[0] * cbi->matrix[11] + cbi->matrix[3];
	zip[k*3+1] = z * (cbi->matrix[6] + cbi->ic[1] * cbi->matrix[10]) 
		+ cbi->ic[1] * cbi->matrix[11] + cbi->matrix[7];
	zip[k*3+2] = z * cbi->matrix[10] + cbi->matrix[11];
    }
    
    /* Main loop */
    p = 0;
    for (k = 0; k < vol->dim[2]; k++) {
	for (j = 0; j < vol->dim[1]; j++) {
	    vec3_add3 (acc2, &zip[3*k], &yip[3*j]);
	    for (i = 0; i < vol->dim[0]; i++) {
		vec3_add3 (acc3, acc2, &xip[3*i]);
		dw = 1 / acc3[2];
		acc3[0] = acc3[0] * dw;
		acc3[1] = acc3[1] * dw;
		img[p++] += dw * dw * get_pixel_value_c (cbi, acc3[0], acc3[1]);
	    }
	}
    }
    free (xip);
    free (yip);
    free (zip);
}

void
project_volume_onto_image_b (Volume* vol, CB_Image* cbi, float scale)
{
    int i, j, k, p;
    float* img = (float*) vol->img;
    double wip[3];
    double *xip, *yip, *zip;
    double acc1[3],acc2[3],acc3[3];
    double *x, *y, *z;
    double dw;
    double sad_sid_2;

    /* Rescale image (destructive rescaling) */
    sad_sid_2 = (cbi->sad * cbi->sad) / (cbi->sid * cbi->sid);
    for (i = 0; i < cbi->dim[0]*cbi->dim[1]; i++) {
	cbi->img[i] *= sad_sid_2;	// Speedup trick re: Kachelsreiss
	cbi->img[i] *= scale;		// User scaling
    }

    x = (double*) malloc (vol->dim[0]*sizeof(double));
    y = (double*) malloc (vol->dim[1]*sizeof(double));
    z = (double*) malloc (vol->dim[2]*sizeof(double));
    xip = (double*) malloc (3*vol->dim[0]*sizeof(double));
    yip = (double*) malloc (3*vol->dim[1]*sizeof(double));
    zip = (double*) malloc (3*vol->dim[2]*sizeof(double));

    /* Precompute partial projections here */
    for (i = 0; i < vol->dim[0]; i++) {
	x[i] = (double) (vol->offset[0] + i * vol->pix_spacing[0]);
	xip[i*3+0] = x[i] * cbi->matrix[0];
	xip[i*3+1] = x[i] * cbi->matrix[4];
	xip[i*3+2] = x[i] * cbi->matrix[8];
	x[i] *= cbi->nrm[0];
    }
    for (j = 0; j < vol->dim[1]; j++) {
	y[j] = (double) (vol->offset[1] + j * vol->pix_spacing[1]);
	yip[j*3+0] = y[j] * cbi->matrix[1];
	yip[j*3+1] = y[j] * cbi->matrix[5];
	yip[j*3+2] = y[j] * cbi->matrix[9];
	y[j] *= cbi->nrm[1];
    }
    for (k = 0; k < vol->dim[2]; k++) {
	z[k] = (double) (vol->offset[2] + k * vol->pix_spacing[2]);
	zip[k*3+0] = z[k] * cbi->matrix[2];
	zip[k*3+1] = z[k] * cbi->matrix[6];
	zip[k*3+2] = z[k] * cbi->matrix[10];
	z[k] *= cbi->nrm[2];
	z[k] = cbi->sad - z[k];
    }
    wip[0] = cbi->matrix[3];
    wip[1] = cbi->matrix[7];
    wip[2] = cbi->matrix[11];
    
    /* Main loop */
    p = 0;
    for (k = 0; k < vol->dim[2]; k++) {
	vec3_add3 (acc1, wip, &zip[3*k]);
	for (j = 0; j < vol->dim[1]; j++) {
	    vec3_add3 (acc2, acc1, &yip[3*j]);
	    for (i = 0; i < vol->dim[0]; i++) {
		vec3_add3 (acc3, acc2, &xip[3*i]);
		dw = 1 / acc3[2];
		acc3[0] = cbi->ic[0] + acc3[0] * dw;
		acc3[1] = cbi->ic[1] + acc3[1] * dw;
		img[p++] += dw * dw * get_pixel_value_c (cbi, acc3[0], acc3[1]);
	    }
	}
    }
    free (x);
    free (y);
    free (z);
    free (xip);
    free (yip);
    free (zip);
}

void
project_volume_onto_image_a (Volume* vol, CB_Image* cbi, float scale)
{
    int i, j, k, p;
    float* img = (float*) vol->img;
    double wip[3];
    double *xip, *yip, *zip;
    double acc1[3],acc2[3],acc3[3];
    double *x, *y, *z;
    double s1, s, sad2;

    /* Rescale image (destructive rescaling) */
    for (i = 0; i < cbi->dim[0]*cbi->dim[1]; i++) {
	cbi->img[i] *= scale;
    }

    x = (double*) malloc (vol->dim[0]*sizeof(double));
    y = (double*) malloc (vol->dim[1]*sizeof(double));
    z = (double*) malloc (vol->dim[2]*sizeof(double));
    xip = (double*) malloc (3*vol->dim[0]*sizeof(double));
    yip = (double*) malloc (3*vol->dim[1]*sizeof(double));
    zip = (double*) malloc (3*vol->dim[2]*sizeof(double));

    /* Precompute partial projections here */
    for (i = 0; i < vol->dim[0]; i++) {
	x[i] = (double) (vol->offset[0] + i * vol->pix_spacing[0]);
	xip[i*3+0] = x[i] * cbi->matrix[0];
	xip[i*3+1] = x[i] * cbi->matrix[4];
	xip[i*3+2] = x[i] * cbi->matrix[8];
	x[i] *= cbi->nrm[0];
    }
    for (j = 0; j < vol->dim[1]; j++) {
	y[j] = (double) (vol->offset[1] + j * vol->pix_spacing[1]);
	yip[j*3+0] = y[j] * cbi->matrix[1];
	yip[j*3+1] = y[j] * cbi->matrix[5];
	yip[j*3+2] = y[j] * cbi->matrix[9];
	y[j] *= cbi->nrm[1];
    }
    for (k = 0; k < vol->dim[2]; k++) {
	z[k] = (double) (vol->offset[2] + k * vol->pix_spacing[2]);
	zip[k*3+0] = z[k] * cbi->matrix[2];
	zip[k*3+1] = z[k] * cbi->matrix[6];
	zip[k*3+2] = z[k] * cbi->matrix[10];
	z[k] *= cbi->nrm[2];
	z[k] = cbi->sad - z[k];
    }
    wip[0] = cbi->matrix[3];
    wip[1] = cbi->matrix[7];
    wip[2] = cbi->matrix[11];
    sad2 = cbi->sad * cbi->sad;
    
    /* Main loop */
    p = 0;
    for (k = 0; k < vol->dim[2]; k++) {
	vec3_add3 (acc1, wip, &zip[3*k]);
	s = z[k];
	for (j = 0; j < vol->dim[1]; j++) {
	    vec3_add3 (acc2, acc1, &yip[3*j]);
	    s1 = z[k] - y[j];
	    for (i = 0; i < vol->dim[0]; i++) {
		s = s1 - x[i];
		//printf ("%10.10g ", s);
		s = sad2 / (s * s);
		vec3_add3 (acc3, acc2, &xip[3*i]);
		//printf ("%10.10g\n", acc3[2]);
		acc3[0] = cbi->ic[0] + acc3[0] / acc3[2];
		acc3[1] = cbi->ic[1] + acc3[1] / acc3[2];
		img[p++] += s * get_pixel_value_b (cbi, acc3[0], acc3[1]);
	    }
	}
    }
    free (x);
    free (y);
    free (z);
    free (xip);
    free (yip);
    free (zip);
}

void
project_volume_onto_image_reference (Volume* vol, CB_Image* cbi, float scale)
{
    int i, j, k, p;
    double vp[4];   /* vp = voxel position */
    float* img = (float*) vol->img;
    
    p = 0;
    vp[3] = 1.0;
    for (k = 0; k < vol->dim[2]; k++) {
	vp[2] = (double) (vol->offset[2] + k * vol->pix_spacing[2]);
	for (j = 0; j < vol->dim[1]; j++) {
	    vp[1] = (double) (vol->offset[1] + j * vol->pix_spacing[1]);
	    for (i = 0; i < vol->dim[0]; i++) {
		double ip[3];        /* ip = image position */
		double s;            /* s = projection of vp onto s axis */
		vp[0] = (double) (vol->offset[0] + i * vol->pix_spacing[0]);
		mat43_mult_vec3 (ip, cbi->matrix, vp);
		ip[0] = cbi->ic[0] + ip[0] / ip[2];
		ip[1] = cbi->ic[1] + ip[1] / ip[2];
		/* Distance on axis from ctr to source */
		s = vec3_dot (cbi->nrm, vp);
		/* Conebeam weighting factor */
		s = cbi->sad - s;
		s = cbi->sad * cbi->sad / (s * s);
		img[p++] += scale * s * get_pixel_value_b (cbi, ip[0], ip[1]);
	    }
	}
    }
}

void
reconstruct_conebeam (Volume* vol, MGHCBCT_Options* options)
{
    int i;
    int num_imgs;
    float scale;

    num_imgs = 1 + (options->last_img - options->first_img)
	    / options->skip_img;

    scale = (float) (sqrt(3) / (double) num_imgs);
    scale = scale * options->scale;

    for (i = options->first_img; i <= options->last_img; i += options->skip_img) {
	CB_Image* cbi;
	cbi = get_image(options, i);
	
	// printf ("Projecting Image %d\n", i);
	// project_volume_onto_image_reference (vol, cbi, scale);
	// project_volume_onto_image_a (vol, cbi, scale);
	// project_volume_onto_image_b (vol, cbi, scale);
	project_volume_onto_image_c (vol, cbi, scale);
	free_cb_image (cbi);
    }
}

int main(int argc, char* argv[])
{
    MGHCBCT_Options options;
    Volume* vol;
    
    parse_args (&options, argc, argv);
    vol = my_create_volume (&options);

    switch (options.threading) {
    case THREADING_CPU:
	reconstruct_conebeam (vol, &options);
	break;
    case THREADING_BROOK:
#if (HAVE_BROOK)
	reconstruct_conebeam (vol, &options);
#else
	reconstruct_conebeam (vol, &options);
#endif
	break;
    case THREADING_CUDA:
#if (CUDA_FOUND)
	CUDA_reconstruct_conebeam (vol, &options);
#else
	reconstruct_conebeam (vol, &options);
#endif
	break;
    }

    convert_to_hu (vol, &options);

    printf ("Writing output volume...\n");
    write_mha (options.output_file, vol);
    printf(" done.\n\n");

    return 0;
}
