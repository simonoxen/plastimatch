/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fdk_opts.h"
#include "fdk_utils.h"
#include "file_util.h"
#include "mathutil.h"
#include "proj_image.h"
#include "ramp_filter.h"
#include "volume.h"

static void
load_pfm (Proj_image *proj, char* img_filename)
{
    FILE* fp;
    char buf[1024];
    size_t rc;

    if (!proj) return;

    fp = fopen (img_filename,"rb");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for read\n", img_filename);
	exit (-1);
    }

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
    if (2 != sscanf (buf, "%d %d", &proj->dim[0], &proj->dim[1])) {
	fprintf (stderr, "Couldn't parse file %s as an image [2]\n", 
		 img_filename);
	exit (-1);
    }
    /* Skip third line */
    fgets (buf, 1024, fp);

    /* Malloc memory */
    proj->img = (float*) malloc (sizeof(float) * proj->dim[0] * proj->dim[1]);
    if (!proj->img) {
	fprintf (stderr, "Couldn't malloc memory for input image\n");
	exit (-1);
    }

    /* Load pixels */
    rc = fread (proj->img, sizeof(float), proj->dim[0] * proj->dim[1], fp);
    if (rc != proj->dim[0] * proj->dim[1]) {
	fprintf (stderr, "Couldn't load raster data for %s\n",
		 img_filename);
	exit (-1);
    }
    fclose (fp);
}

static void
load_mat (Proj_image *proj, char* mat_filename)
{
    FILE* fp;
    int i;
    float f;

    if (!proj) return;

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
	proj->ic[i] = (double) f;
    }
    /* Load projection matrix */
    for (i = 0; i < 12; i++) {
	if (1 != fscanf (fp, "%g", &f)) {
	    fprintf (stderr, "Couldn't parse file %s as a matrix [2,%d]\n", 
		     mat_filename, i);
	    exit (-1);
	}
	proj->matrix[i] = (double) f;
    }
    /* Load sad */
    if (1 != fscanf (fp, "%g", &f)) {
	fprintf (stderr, "Couldn't load sad from %s\n", mat_filename);
	exit (-1);
    }
    proj->sad = (double) f;
    /* Load sid */
    if (1 != fscanf (fp, "%g", &f)) {
	fprintf (stderr, "Couldn't load sad from %s\n", mat_filename);
	exit (-1);
    }
    proj->sid = (double) f;
    /* Load nrm vector */
    for (i = 0; i < 3; i++) {
	if (1 != fscanf (fp, "%g", &f)) {
	    fprintf (stderr, "Couldn't parse file %s as a matrix [1,%d]\n", 
		     mat_filename, i);
	    exit (-1);
	}
	proj->nrm[i] = (double) f;
    }
    fclose (fp);
}

void
proj_image_initialize (Proj_image *proj)
{
    memset (proj, 0, sizeof(Proj_image));
}

Proj_image*
proj_image_create (void)
{
    Proj_image *proj;

    proj = (Proj_image*) malloc (sizeof(Proj_image));
    if (!proj) return 0;

    proj_image_initialize (proj);

    return proj;
}

void
proj_image_debug_header (Proj_image *proj)
{
    int i;
    printf ("Image center: %g %g\n", proj->ic[0], proj->ic[1]);
    printf ("Projection matrix: ");
    for (i = 0; i < 12; i++) {
	printf ("%g ", proj->matrix[i]);
    }
    printf ("\n");
}

void
proj_image_stats (Proj_image *proj)
{
    int i, num;
    float min_val, max_val;
    double sum = 0.0;

    if (!proj) {
	printf ("No image.\n");
	return;
    }

    num = proj->dim[0]*proj->dim[1];
    if (!proj->img || num == 0) {
	printf ("No image.\n");
	return;
    }
    
    min_val = max_val = proj->img[0];
    for (i = 0; i < num; i++) {
	float v = proj->img[i];
	if (min_val > v) min_val = v;
	if (max_val < v) max_val = v;
	sum += v;
    }

    printf ("MIN %f AVE %f MAX %f NUM %d\n",
	    min_val, (float) (sum / num), max_val, num);
}

Proj_image* 
proj_image_load_pfm (char* img_filename, char* mat_filename)
{
    Proj_image* proj;

    if (!img_filename) return 0;

    proj = proj_image_create ();
    if (!proj) return 0;

    load_pfm (proj, img_filename);

    if (mat_filename) {
	load_mat (proj, mat_filename);
    } else {
	/* No mat file, so try to find automatically */
	int img_filename_len = strlen (img_filename);
	if (img_filename_len > 4 
	    && !strcmp (&img_filename[img_filename_len-4], ".pfm")) 
	{
	    char *mat_fn = strdup (img_filename);
	    strcpy (&mat_fn[img_filename_len-4], ".txt");
	    if (file_exists (mat_fn)) {
		load_mat (proj, mat_fn);
	    }
	    free (mat_fn);
	}
    }

    return proj;
}

Proj_image* 
proj_image_load_and_filter (
    Fdk_options * options, 
    char* img_filename, 
    char* mat_filename
)
{
    int i,j;
    size_t rc;
    float f;
    FILE* fp;

    Proj_image* proj;
    unsigned short * readimg;
    int movelength,fillhead,filltail;

    fillhead=512;
    filltail=55;
    movelength=(512-filltail);

    fp = fopen (img_filename,"rb");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for read\n. Skipped", 
	    img_filename);
	return NULL;
    }

    proj = (Proj_image*) malloc (sizeof(Proj_image));

    //only support 512x384
    if (options->full_fan) {
	proj->dim[0]=384;
	proj->dim[1]=512;
    }
    else {
	proj->dim[0]=384;
	proj->dim[1]=1024;
    }
    /* Malloc memory */
    proj->img = (float*) malloc (sizeof(float) * proj->dim[0] * proj->dim[1]);
    if (!proj->img) {
	fprintf (stderr, "Couldn't malloc memory for input image\n");
	exit (-1);
    }
    memset(proj->img,0,proj->dim[0] * proj->dim[1]*sizeof(float));
	
    readimg = (unsigned short*) malloc (sizeof(unsigned short) * 512 * proj->dim[0]);
    if (!readimg ) {
	fprintf (stderr, "Couldn't malloc memory for input image\n");
	exit (-1);
    }

    /* Load pixels */
    rc = fread (readimg , sizeof(unsigned short),  512* proj->dim[0], fp);
    if (rc != 512 * proj->dim[0]) {
	fprintf (stderr, "Couldn't load raster data for %s\n",
		 img_filename);
	exit (-1);
    }

#if (FFTW_FOUND)
    RampFilter(readimg,proj->img,512,proj->dim[0]);
#endif

    free(readimg);

    if(!options->full_fan){

	//ImageView imgview(IF_FLOAT_32_GREY, 1024, proj->dim[1], proj->img);
	//system("pause");
	for (i=proj->dim[0]-1; i>=0; i--)
	    memcpy(proj->img+1024*i+512-65, proj->img+512*i, 512*sizeof(float));
	for (i=proj->dim[0]-1; i>=0; i--){
	    memset(proj->img+1024*i,0,(fillhead-filltail)*sizeof(float));
	    memset(proj->img+1024*i+1023-65,0,65*sizeof(float));
	}
	for (j=proj->dim[0]-1; j>=0; j--)
	    for(i=(512-filltail);i<=512+filltail-1;i++)
		proj->img[j*1024+i]*=(float)(i-(512-filltail-1))/(float)(512+filltail-1+1-(512-filltail-1));

    }

	
    //ImageView imgview(IF_FLOAT_32_GREY, proj->dim[0], proj->dim[1], proj->img);
    //   system("pause");
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
	proj->ic[i] = (double) f;
    }
    /* Load projection matrix */
    for (i = 0; i < 12; i++) {
	if (1 != fscanf (fp, "%g", &f)) {
	    fprintf (stderr, "Couldn't parse file %s as a matrix [2,%d]\n", 
		     mat_filename, i);
	    exit (-1);
	}
	proj->matrix[i] = (double) f;
    }
    /* Load sad */
    if (1 != fscanf (fp, "%g", &f)) {
	fprintf (stderr, "Couldn't load sad from %s\n", mat_filename);
	exit (-1);
    }
    proj->sad = (double) f;
    /* Load sid */
    if (1 != fscanf (fp, "%g", &f)) {
	fprintf (stderr, "Couldn't load sad from %s\n", mat_filename);
	exit (-1);
    }
    proj->sid = (double) f;
    /* Load nrm vector */
    for (i = 0; i < 3; i++) {
	if (1 != fscanf (fp, "%g", &f)) {
	    fprintf (stderr, "Couldn't parse file %s as a matrix [1,%d]\n", 
		     mat_filename, i);
	    exit (-1);
	}
	proj->nrm[i] = (double) f;
    }
    fclose (fp);

#if defined (commentout)
    printf ("Image center: ");
    rawvec2_print_eol (stdout, proj->ic);
    printf ("Projection matrix:\n");
    matrix_print_eol (stdout, proj->matrix, 3, 4);
#endif

    return proj;
}

#if defined (commentout)
Proj_image*
get_image (Fdk_options* options, int image_num)
{
    char* img_file_pat = "out_%04d.pfm";
    char* mat_file_pat = "out_%04d.txt";

    char img_file[1024], mat_file[1024], fmt[1024];
    sprintf (fmt, "%s/%s", options->input_dir, img_file_pat);
    sprintf (img_file, fmt, image_num);
    sprintf (fmt, "%s/%s", options->input_dir, mat_file_pat);
    sprintf (mat_file, fmt, image_num);
    return load_cb_image (img_file, mat_file);
}
#endif

void
proj_image_free (Proj_image* proj)
{
    free (proj->img);
    free (proj);
}
