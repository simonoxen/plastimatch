/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fdk.h"
#include "fdk_opts.h"
#include "fdk_utils.h"
#include "proj_image.h"
#include "ramp_filter.h"
#include "volume.h"

CB_Image* 
proj_image_load_pfm (char* img_filename, char* mat_filename)
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

CB_Image* 
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

    CB_Image* cbi;
    unsigned short * readimg;
    int movelength,fillhead,filltail;

    fillhead=512;
    filltail=55;
    movelength=(512-filltail);

    cbi = (CB_Image*) malloc (sizeof(CB_Image));

    fp = fopen (img_filename,"rb");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for read\n. Skipped", img_filename);
	return NULL;
    }

    //only support 512x384
    if (options->full_fan) {
	cbi->dim[0]=384;
	cbi->dim[1]=512;
    }
    else {
	cbi->dim[0]=384;
	cbi->dim[1]=1024;
    }
    /* Malloc memory */
    cbi->img = (float*) malloc (sizeof(float) * cbi->dim[0] * cbi->dim[1]);
    if (!cbi->img) {
	fprintf (stderr, "Couldn't malloc memory for input image\n");
	exit (-1);
    }
    memset(cbi->img,0,cbi->dim[0] * cbi->dim[1]*sizeof(float));
	
    readimg = (unsigned short*) malloc (sizeof(unsigned short) * 512 * cbi->dim[0]);
    if (!readimg ) {
	fprintf (stderr, "Couldn't malloc memory for input image\n");
	exit (-1);
    }

    /* Load pixels */
    rc = fread (readimg , sizeof(unsigned short),  512* cbi->dim[0], fp);
    if (rc != 512 * cbi->dim[0]) {
	fprintf (stderr, "Couldn't load raster data for %s\n",
		 img_filename);
	exit (-1);
    }

#if (FFTW_FOUND)
    RampFilter(readimg,cbi->img,512,cbi->dim[0]);
#endif

    free(readimg);

    if(!options->full_fan){

	//ImageView imgview(IF_FLOAT_32_GREY, 1024, cbi->dim[1], cbi->img);
	//system("pause");
	for (i=cbi->dim[0]-1; i>=0; i--)
	    memcpy(cbi->img+1024*i+512-65, cbi->img+512*i, 512*sizeof(float));
	for (i=cbi->dim[0]-1; i>=0; i--){
	    memset(cbi->img+1024*i,0,(fillhead-filltail)*sizeof(float));
	    memset(cbi->img+1024*i+1023-65,0,65*sizeof(float));
	}
	for (j=cbi->dim[0]-1; j>=0; j--)
	    for(i=(512-filltail);i<=512+filltail-1;i++)
		cbi->img[j*1024+i]*=(float)(i-(512-filltail-1))/(float)(512+filltail-1+1-(512-filltail-1));

    }

	
    //ImageView imgview(IF_FLOAT_32_GREY, cbi->dim[0], cbi->dim[1], cbi->img);
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

#if defined (commentout)
CB_Image*
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
free_cb_image (CB_Image* cbi)
{
    free (cbi->img);
    free (cbi);
}
