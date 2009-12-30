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
#include "hnd_io.h"
#include "mathutil.h"
#include "proj_image.h"
#include "ramp_filter.h"
#include "volume.h"

/* -----------------------------------------------------------------------
   Private functions
   ----------------------------------------------------------------------- */
static void
raw_load (Proj_image *proj, char* img_filename)
{
    FILE* fp;
    size_t rc;
    uint64_t fs;

    if (!proj) return;

    /* Open file */
    fp = fopen (img_filename,"rb");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for read\n", img_filename);
	exit (-1);
    }
    
    /* Malloc memory */
    fs = file_size (img_filename);
    proj->img = (float*) malloc (fs);
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

    /* Guess image size */
    switch (fs) {
    case (512*384*sizeof(float)):
	proj->dim[0] = 512;
	proj->dim[1] = 384;
	break;
    case (1024*384*sizeof(float)):
	proj->dim[0] = 1024;
	proj->dim[1] = 384;
	break;
    case (1024*768*sizeof(float)):
	proj->dim[0] = 1024;
	proj->dim[1] = 768;
	break;
    case (2048*1536*sizeof(float)):
	proj->dim[0] = 1024;
	proj->dim[1] = 768;
	break;
    default:
	proj->dim[0] = 1024;
	proj->dim[1] = fs / (1024 * sizeof(float));
	break;
    }
}

static void
raw_save (Proj_image *proj, char* img_filename)
{
    FILE* fp;
    
    fp = fopen (img_filename, "wb");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for write\n", img_filename);
	exit (-1);
    }

    fwrite (proj->img, sizeof(float), proj->dim[0]*proj->dim[1], fp);
    fclose (fp);
}

static void
pfm_load (Proj_image *proj, char* img_filename)
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
pfm_save (Proj_image *proj, char* img_filename)
{
    FILE* fp;
    
    make_directory_recursive (img_filename);
    fp = fopen (img_filename, "wb");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for write\n", img_filename);
	exit (-1);
    }

    fprintf (fp, 
	"Pf\n"
	"%d %d\n"
	"-1\n",
	proj->dim[0], proj->dim[1]);

    fwrite (proj->img, sizeof(float), proj->dim[0]*proj->dim[1], fp);
    fclose (fp);
}

static void
pgm_save (Proj_image *proj, char* img_filename)
{
    FILE* fp;
    int i;
    
    make_directory_recursive (img_filename);
    fp = fopen (img_filename, "wb");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for write\n", img_filename);
	exit (-1);
    }

    fprintf (fp, 
	"P2\n"
	"# Created by plastimatch\n"
	"%d %d\n"
	"65535\n",
	proj->dim[0], proj->dim[1]);

    for (i = 0; i < proj->dim[0]*proj->dim[1]; i++) {
	float v = proj->img[i];
	if (v > 65536) {
	    v = 65536;
	} else if (v < 0) {
	    v = 0;
	}
	fprintf (fp,"%lu ", ROUND_INT(v));
	if (i % 25 == 24) {
	    fprintf (fp,"\n");
	}
    }
    fclose (fp);
}

static void
mat_load (Proj_image *proj, char* mat_filename)
{
    FILE* fp;
    int i;
    float f;
    Proj_matrix *pmat;

    if (!proj) return;

    /* Allocate memory */
    pmat = proj_matrix_create ();

    /* Open file */
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
	pmat->ic[i] = (double) f;
    }
    /* Load projection matrix */
    for (i = 0; i < 12; i++) {
	if (1 != fscanf (fp, "%g", &f)) {
	    fprintf (stderr, "Couldn't parse file %s as a matrix [2,%d]\n", 
		     mat_filename, i);
	    exit (-1);
	}
	pmat->matrix[i] = (double) f;
    }
    /* Load sad */
    if (1 != fscanf (fp, "%g", &f)) {
	fprintf (stderr, "Couldn't load sad from %s\n", mat_filename);
	exit (-1);
    }
    pmat->sad = (double) f;
    /* Load sid */
    if (1 != fscanf (fp, "%g", &f)) {
	fprintf (stderr, "Couldn't load sad from %s\n", mat_filename);
	exit (-1);
    }
    pmat->sid = (double) f;
    /* Load nrm vector */
    for (i = 0; i < 3; i++) {
	if (1 != fscanf (fp, "%g", &f)) {
	    fprintf (stderr, "Couldn't parse file %s as a matrix [1,%d]\n", 
		     mat_filename, i);
	    exit (-1);
	}
	pmat->nrm[i] = (double) f;
    }
    fclose (fp);

    proj->pmat = pmat;
}

static void
mat_load_by_img_filename (Proj_image* proj, char* img_filename)
{
    /* No mat file, so try to find automatically */
    int img_filename_len = strlen (img_filename);
    if (img_filename_len > 4) {
	char *mat_fn = strdup (img_filename);
	strcpy (&mat_fn[img_filename_len-4], ".txt");
	if (file_exists (mat_fn)) {
	    mat_load (proj, mat_fn);
	}
	free (mat_fn);
    }
}

static Proj_image* 
proj_image_load_pfm (char* img_filename, char* mat_filename)
{
    Proj_image* proj;

    if (!img_filename) return 0;

    proj = proj_image_create ();
    if (!proj) return 0;

    pfm_load (proj, img_filename);

    if (mat_filename) {
	mat_load (proj, mat_filename);
    } else {
	mat_load_by_img_filename (proj, img_filename);
    }

    return proj;
}

static Proj_image* 
proj_image_load_raw (char* img_filename, char* mat_filename)
{
    Proj_image* proj;

    if (!img_filename) return 0;

    proj = proj_image_create ();
    if (!proj) return 0;

    raw_load (proj, img_filename);

    if (mat_filename) {
	mat_load (proj, mat_filename);
    } else {
	mat_load_by_img_filename (proj, img_filename);
    }

    return proj;
}

static Proj_image* 
proj_image_load_hnd (char* img_filename)
{
    Proj_image* proj;

    if (!img_filename) return 0;

    proj = proj_image_create ();
    if (!proj) return 0;

    hnd_load (proj, img_filename);
    if (proj->img == 0) {
	proj_image_destroy (proj);
	return 0;
    }

    return proj;
}

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
void
proj_image_init (Proj_image *proj)
{
    memset (proj, 0, sizeof(Proj_image));
}

Proj_image*
proj_image_create (void)
{
    Proj_image *proj;

    proj = (Proj_image*) malloc (sizeof(Proj_image));
    if (!proj) return 0;

    proj_image_init (proj);

    return proj;
}

void
proj_image_create_pmat (Proj_image *proj)
{
    /* Allocate memory */
    proj->pmat = proj_matrix_create ();
}

void
proj_image_create_img (Proj_image *proj, int dim[2])
{
    proj->dim[0] = dim[0];
    proj->dim[1] = dim[1];
    proj->img = (float*) malloc (sizeof(float) * proj->dim[0] * proj->dim[1]);
}

void
proj_image_debug_header (Proj_image *proj)
{
    int i;
    printf ("Image center: %g %g\n", proj->pmat->ic[0], proj->pmat->ic[1]);
    printf ("Projection matrix: ");
    for (i = 0; i < 12; i++) {
	printf ("%g ", proj->pmat->matrix[i]);
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
proj_image_load (
    char* img_filename,
    char* mat_filename
)
{
    if (extension_is (img_filename, ".pfm")) {
	return proj_image_load_pfm (img_filename, mat_filename);
    }
    else if (extension_is (img_filename, ".raw")) {
	return proj_image_load_raw (img_filename, mat_filename);
    }
    else if (extension_is (img_filename, ".hnd")) {
	return proj_image_load_hnd (img_filename);
    }
    return 0;
}

void
proj_image_save (
    Proj_image *proj,
    char *img_filename,
    char *mat_filename
)
{
    if (extension_is (img_filename, ".pfm")) {
	pfm_save (proj, img_filename);
    }
    else if (extension_is (img_filename, ".raw")) {
	raw_save (proj, img_filename);
    }
    else if (extension_is (img_filename, ".pgm")) {
	pgm_save (proj, img_filename);
    }

    proj_matrix_save (proj->pmat, mat_filename);
}

void
proj_image_filter (Proj_image *proj)
{
#if (FFTW_FOUND)
    ramp_filter (proj->img, proj->dim[0], proj->dim[1]);
#endif
}

#if defined (commentout)
Proj_image* 
proj_image_load_and_filter (
    Fdk_options * options, 
    char* img_filename, 
    char* mat_filename
)
{
    Proj_image* proj = 0;

#if defined (commentout)
    int i,j;
    size_t rc;
    float f;
    FILE* fp;

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
    memset (proj->img, 0, proj->dim[0] * proj->dim[1] * sizeof(float));
	
    readimg = (unsigned short*) malloc (sizeof(unsigned short) * 512 * proj->dim[0]);
    if (!readimg) {
	fprintf (stderr, "Couldn't malloc memory for input image\n");
	exit (-1);
    }

    /* Load pixels */
    rc = fread (readimg, sizeof(unsigned short), 512 * proj->dim[0], fp);
    if (rc != 512 * proj->dim[0]) {
	fprintf (stderr, "Couldn't load raster data for %s\n",
		 img_filename);
	exit (-1);
    }

#if (FFTW_FOUND)
    RampFilter(readimg,proj->img,512,proj->dim[0]);
#endif

    free (readimg);

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

#endif

    return proj;
}
#endif

void
proj_image_free (Proj_image* proj)
{
    if (!proj) return;
    if (proj->pmat) {
	free (proj->pmat);
    }
    if (proj->img) {
	free (proj->img);
    }
}

void
proj_image_destroy (Proj_image* proj)
{
    proj_image_free (proj);
    free (proj);
}
