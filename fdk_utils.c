/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "fdk.h"
#include "fdk_opts.h"
#include "volume.h"

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

#define READ_PFM 1

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
	fprintf (stderr, "Couldn't load sad from %s\n");
	exit (-1);
    }
    cbi->sad = (double) f;
    /* Load sid */
    if (1 != fscanf (fp, "%g", &f)) {
	fprintf (stderr, "Couldn't load sad from %s\n");
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

CB_Image*
get_image (MGHCBCT_Options* options, int image_num)
{
#if defined (READ_PFM)
    char* img_file_pat = "out_%04d.pfm";
#else
    char* img_file_pat = "out_%04d.pgm";
#endif
    char* mat_file_pat = "out_%04d.txt";

    char img_file[1024], mat_file[1024], fmt[1024];
    sprintf (fmt, "%s/%s", options->input_dir, img_file_pat);
    sprintf (img_file, fmt, image_num);
    sprintf (fmt, "%s/%s", options->input_dir, mat_file_pat);
    sprintf (mat_file, fmt, image_num);
    return load_cb_image (img_file, mat_file);
}

void
free_cb_image (CB_Image* cbi)
{
    free (cbi->img);
    free (cbi);
}

