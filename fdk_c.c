/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "mathutil.h"
#include "fdk.h"
#include "readmha.h"
#include "fdk_opts.h"

#define READ_PFM

Volume*
my_create_volume (MGHCBCT_Options* options)
{
    float offset[3];
    float spacing[3];
    float* vol_size = options->vol_size;
    int* resolution = options->resolution;

    spacing[0] = vol_size[0] / resolution[0];
    spacing[1] = vol_size[1] / resolution[1];
    spacing[2] = vol_size[2] / resolution[2];

    offset[0] = -vol_size[0] / 2.0f + spacing[0] / 2.0f;
    offset[1] = -vol_size[1] / 2.0f + spacing[1] / 2.0f;
    offset[2] = -vol_size[2] / 2.0f + spacing[2] / 2.0f;

    return volume_create (resolution, offset, spacing, PT_FLOAT, 0);
}

CB_Image* 
load_cb_image (char* img_filename, char* mat_filename)
{
    int i;
    size_t rc;
    float f;
    FILE* fp;
    char buf[1024];
    CB_Image* cbi;

    cbi = (CB_Image*) malloc (sizeof(CB_Image));

    fp = fopen (img_filename,"rb");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for read\n", img_filename);
	exit (-1);
    }

#if defined (READ_PFM)
    /* Verify that it is pfm */
    fgets (buf, 1024, fp);
    if (strncmp(buf, "Pf", 2)) {
	fprintf (stderr, "Couldn't parse file %s as an image [1]\n",
		 img_filename);
	printf (buf);
	exit (-1);
    }
    /* Get image resolution */
    fgets (buf, 1024, fp);
    if (2 != sscanf (buf, "%d %d", &cbi->dim[0], &cbi->dim[1])) {
	fprintf (stderr, "Couldn't parse file %s as an image [2]\n", 
		 img_filename);
	exit (-1);
    }
    /* Skip third line */
    fgets (buf, 1024, fp);

    /* Malloc memory */
    cbi->img = (float*) malloc (sizeof(float) * cbi->dim[0] * cbi->dim[1]);
    if (!cbi->img) {
	fprintf (stderr, "Couldn't malloc memory for input image\n");
	exit (-1);
    }

    /* Load pixels */
    rc = fread (cbi->img, sizeof(float), cbi->dim[0] * cbi->dim[1], fp);
    if (rc != cbi->dim[0] * cbi->dim[1]) {
	fprintf (stderr, "Couldn't load raster data for %s\n",
		 img_filename);
	exit (-1);
    }
#else
    /* Verify that it is pgm */
    fgets (buf, 1024, fp);
    if (strncmp(buf, "P2", 2)) {
	fprintf (stderr, "Couldn't parse file %s as an image [1]\n",
		 img_filename);
	printf (buf);
	exit (-1);
    }
    /* Skip comment line */
    fgets (buf, 1024, fp);
    /* Get image resolution */
    fgets (buf, 1024, fp);
    if (2 != sscanf (buf, "%d %d", &cbi->dim[0], &cbi->dim[1])) {
	fprintf (stderr, "Couldn't parse file %s as an image [2]\n", 
		 img_filename);
	exit (-1);
    }
    /* Skip max gray */
    fgets (buf, 1024, fp);

    /* Malloc memory */
    cbi->img = (float*) malloc (sizeof(float) * cbi->dim[0] * cbi->dim[1]);
    if (!cbi->img) {
	fprintf (stderr, "Couldn't malloc memory for input image\n");
	exit (-1);
    }

    /* Load pixels */
    for (i = 0; i < cbi->dim[0] * cbi->dim[1]; i++) {
	if (1 != fscanf (fp, "%g", &cbi->img[i])) {
	    fprintf (stderr, "Couldn't parse file %s as an image [3,%d]\n", 
		     img_filename, i);
	    exit (-1);
	}
    }
#endif
    fclose (fp);

    /* Load projection matrix */
    fp = fopen (mat_filename,"r");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for read\n", mat_filename);
	exit (-1);
    }
    /* Load image center */
    for (i = 0; i < 2; i++) {
	if (1 != fscanf (fp, "%g", &f)) {
	    fprintf (stderr, "Couldn't parse file %s as a matrix [1,%d]\n", 
		     mat_filename, i);
	    exit (-1);
	}
	cbi->ic[i] = (double) f;
    }
    /* Load projection matrix */
    for (i = 0; i < 12; i++) {
	if (1 != fscanf (fp, "%g", &f)) {
	    fprintf (stderr, "Couldn't parse file %s as a matrix [2,%d]\n", 
		     mat_filename, i);
	    exit (-1);
	}
	cbi->matrix[i] = (double) f;
    }
    /* Load sad */
    if (1 != fscanf (fp, "%g", &f)) {
	fprintf (stderr, "Couldn't load sad from %s\n", mat_filename);
	exit (-1);
    }
    cbi->sad = (double) f;
    /* Load sid */
    if (1 != fscanf (fp, "%g", &f)) {
	fprintf (stderr, "Couldn't load sad from %s\n", mat_filename);
	exit (-1);
    }
    cbi->sid = (double) f;
    /* Load nrm vector */
    for (i = 0; i < 3; i++) {
	if (1 != fscanf (fp, "%g", &f)) {
	    fprintf (stderr, "Couldn't parse file %s as a matrix [1,%d]\n", 
		     mat_filename, i);
	    exit (-1);
	}
	cbi->nrm[i] = (double) f;
    }
    fclose (fp);

#if defined (commentout)
    printf ("Image center: ");
    rawvec2_print_eol (stdout, cbi->ic);
    printf ("Projection matrix:\n");
    matrix_print_eol (stdout, cbi->matrix, 3, 4);
#endif

    return cbi;
}

void
free_cb_image (CB_Image* cbi)
{
    free (cbi->img);
    free (cbi);
}

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
#if defined (READ_PFM)
    char* img_file_pat = "out_%04d.pfm";
#else
    char* img_file_pat = "out_%04d.pgm";
#endif
    char* mat_file_pat = "out_%04d.txt";

    int num_imgs = 1 + (options->last_img - options->first_img)
	    / options->skip_img;

    float scale = (float) (sqrt(3) / (double) num_imgs);
    scale = scale * options->scale;
    //  scale = scale * 100;
    //  scale = scale / 100.0;

    for (i = options->first_img; i <= options->last_img; i += options->skip_img) {
	char img_file[1024], mat_file[1024], fmt[1024];
	CB_Image* cbi;
	sprintf (fmt, "%s/%s", options->input_dir, img_file_pat);
	sprintf (img_file, fmt, i);
	sprintf (fmt, "%s/%s", options->input_dir, mat_file_pat);
	sprintf (mat_file, fmt, i);
	printf ("Loading Image %d\n", i);
	cbi = load_cb_image (img_file, mat_file);

	printf ("Projecting Image %d\n", i);
	// project_volume_onto_image_reference (vol, cbi, scale);
	// project_volume_onto_image_a (vol, cbi, scale);
	// project_volume_onto_image_b (vol, cbi, scale);
	project_volume_onto_image_c (vol, cbi, scale);
	free_cb_image (cbi);
    }
}

float
convert_to_hu_pixel (float in_value)
{
    float hu;
    float diameter = 40.0;  /* reconstruction diameter in cm */
    hu = 1000 * ((in_value / diameter) - .167) / .167;
    return hu;
}

void
convert_to_hu (Volume* vol, MGHCBCT_Options* options)
{
    int i, j, k, p;
    float* img = (float*) vol->img;
    
    p = 0;
    for (k = 0; k < vol->dim[2]; k++) {
	for (j = 0; j < vol->dim[1]; j++) {
	    for (i = 0; i < vol->dim[0]; i++) {
		img[p] = convert_to_hu_pixel (img[p]);
		p++;
	    }
	}
    }
}

int main(int argc, char* argv[])
{
    MGHCBCT_Options options;
    Volume* vol;
    
    parse_args (&options, argc, argv);

    vol = my_create_volume (&options);

    reconstruct_conebeam (vol, &options);

    convert_to_hu (vol, &options);

    printf ("Writing output volume...\n");
    write_mha (options.output_file, vol);

    return 0;
}
