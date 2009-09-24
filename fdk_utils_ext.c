/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */

#include "ramp_filter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "fdk.h"
#include "fdk_opts_ext.h"
#include "volume.h"
//#include "imageview.h"
#include "readmha_ext.h"
#include "fftw3.h"
#include "fdk_utils2.h"
//#include "fftw3.h"

//#pragma comment (lib, "C:\AAAfiles\Anthony\FFTW\libfftw3-3.lib")

//extern "C"
//{

Volume*
my_create_volume (MGHCBCT_Options_ext* options)
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

    return volume_create (resolution, offset, spacing, PT_FLOAT, 0, 0);
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
convert_to_hu (Volume* vol, MGHCBCT_Options_ext* options)
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
load_and_filter_cb_image (MGHCBCT_Options_ext * options, char* img_filename, char* mat_filename)
{
    int i,j;
    size_t rc;
    float f;
    FILE* fp;
    char buf[1024];
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

#if defined (READ_PFM)
#if 0
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
#endif
	//only support 512x384


	if(options->full_fan){
		cbi->dim[0]=512;
		cbi->dim[1]=384;
	}
	else{
		cbi->dim[0]=1024;
		cbi->dim[1]=384;
	}
	/* Malloc memory */
	cbi->img = (float*) malloc (sizeof(float) * cbi->dim[0] * cbi->dim[1]);
	if (!cbi->img) {
			fprintf (stderr, "Couldn't malloc memory for input image\n");
			exit (-1);
		}
	memset(cbi->img,0,cbi->dim[0] * cbi->dim[1]*sizeof(float));
	
	readimg = (unsigned short*) malloc (sizeof(unsigned short) * 512 * cbi->dim[1]);
	if (!readimg ) {
			fprintf (stderr, "Couldn't malloc memory for input image\n");
			exit (-1);
		}

		/* Load pixels */
		rc = fread (readimg , sizeof(unsigned short),  512* cbi->dim[1], fp);
		if (rc != 512 * cbi->dim[1]) {
			fprintf (stderr, "Couldn't load raster data for %s\n",
				img_filename);
			exit (-1);
		}

	
		RampFilter(readimg,cbi->img,512,cbi->dim[1]);
			
		free(readimg);

	if(!options->full_fan){

		//ImageView imgview(IF_FLOAT_32_GREY, 1024, cbi->dim[1], cbi->img);
		//system("pause");
#if 0
		for (i=cbi->dim[1]-1; i>=0; i--)
			memcpy(cbi->img+1024*i+512, cbi->img+512*i+65, movelength*sizeof(float));
		for (i=cbi->dim[1]-1; i>=0; i--){
			memset(cbi->img+1024*i,0,fillhead*sizeof(float));
			memset(cbi->img+1024*i+1023-65,0,filltail*sizeof(float));
		}
#endif
#if 1
		for (i=cbi->dim[1]-1; i>=0; i--)
			memcpy(cbi->img+1024*i+512-65, cbi->img+512*i, 512*sizeof(float));
		for (i=cbi->dim[1]-1; i>=0; i--){
			memset(cbi->img+1024*i,0,(fillhead-filltail)*sizeof(float));
			memset(cbi->img+1024*i+1023-65,0,65*sizeof(float));
		}
		for (j=cbi->dim[1]-1; j>=0; j--)
			for(i=(512-filltail);i<=512+filltail-1;i++)
				cbi->img[j*1024+i]*=(float)(i-(512-filltail-1))/(float)(512+filltail-1+1-(512-filltail-1));

#endif

	}

	
	//ImageView imgview(IF_FLOAT_32_GREY, cbi->dim[0], cbi->dim[1], cbi->img);
 //   system("pause");
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

CB_Image*
get_image (MGHCBCT_Options_ext* options, int image_num)
{
#if defined (READ_PFM)
    char* img_file_pat = "Proj_%03d.raw";
//	char* img_file_pat = "out_%04d.pfm";
#else
    char* img_file_pat = "out_%04d.pgm";
#endif
    char* mat_file_pat = "";

    char img_file[1024], mat_file[1024], fmt[1024];
    //sprintf (fmt, "%s\\%s\\%s", options->input_dir,options->sub_dir,img_file_pat);
	//sprintf (fmt, "%s\\%s", options->input_dir,img_file_pat);
 //   sprintf (img_file, fmt, image_num);
 //   sprintf (fmt, "%s\\%s", options->input_dir, mat_file_pat);
 //   sprintf (mat_file, fmt, image_num);
 //   return load_and_filter_cb_image (options,img_file, mat_file);
sprintf (img_file, "%s\\Proj_%03d.raw", options->input_dir,image_num);
 //   sprintf (img_file, fmt, image_num);
 //   sprintf (fmt, "%s\\%s", options->input_dir, mat_file_pat);
 sprintf (mat_file, "%s\\tmp\\out_%04d.txt",options->input_dir, image_num);
    return load_and_filter_cb_image (options,img_file, mat_file);
}

void
free_cb_image (CB_Image* cbi)
{
    free (cbi->img);
    free (cbi);
}

void LowPass(int n, double * v)
{
	int i;
	fftw_complex* in, * fft, * ifft;
	fftw_plan fftp, ifftp;
	in    = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
	fft   = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
	ifft  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
	for( i=0; i<n; i++){
		in[i][0]=v[i];
		in[i][1]=0.0;
	}
	fftp  = fftw_plan_dft_1d(n, in, fft, FFTW_FORWARD, FFTW_ESTIMATE);
	ifftp = fftw_plan_dft_1d(n, fft, ifft, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(fftp);
	for(i=0;i<n;i++){
		double tmp=	pow((cos(i * DEGTORAD * 360/n)+1)/2,20);
		fft[i][0]*=tmp;
		fft[i][1]*=tmp;
	}	
	fftw_execute(ifftp);
	for( i=0; i<n; i++)
		v[i]=ifft[i][0]/(float)n;
}


void process_norm_CBCT(Volume * norm_CBCT, MGHCBCT_Options_ext* options)
{
	int ni,nj,nk;
	double norm_radius;
	double radius;
	int radius_r;
	double average;
	double *average_r;
	int *pixels_r;
	int pixels;
	float *norm;
	average_r=(double *)malloc(norm_CBCT->dim[0]*sizeof(double));
	if (average_r==NULL){
		printf("Malloc error for *average");
		exit(1);
	}
	
	for(ni=0; ni<norm_CBCT->dim[0]; ni++)
		average_r[ni]=0.0f;

	pixels_r=(int *)malloc(norm_CBCT->dim[0]*sizeof(int));
	if (pixels_r==NULL){
		printf("Malloc error for *pixels_r");
		exit(1);
	}

	for(ni=0; ni<norm_CBCT->dim[0]; ni++)
		pixels_r[ni]=0.0f;
	
	pixels=0;
	average=0;

	norm=(float *)norm_CBCT->img;


	if (options->full_fan){
		printf("Processing %s\n",options->Full_normCBCT_name);
		norm_radius=options->Full_radius;
	}
	else{
		printf("Processing %s\n",options->Half_normCBCT_name);
		norm_radius=options->Half_radius;
	}

	for (nj=0; nj<norm_CBCT->dim[1]; nj++){
		for (ni=0; ni<norm_CBCT->dim[0]; ni++){
			float x=(float)ni*norm_CBCT->pix_spacing[0]+norm_CBCT->offset[0]-norm_CBCT->pix_spacing[0];
			float y=(float)nj*norm_CBCT->pix_spacing[1]+norm_CBCT->offset[1]-norm_CBCT->pix_spacing[1];
			radius=sqrt(x*x+y*y);
			radius_r=(int)(radius/norm_CBCT->pix_spacing[0]+0.5);
			for (nk=0; nk<norm_CBCT->dim[2]; nk++){
				if (radius>norm_radius-20 && radius<norm_radius) {
					int pi=volume_index(norm_CBCT->dim,ni,nj,nk);
					float *pp=(float*)norm_CBCT->img+pi;
					average+=norm[pi];
					pixels++;
				}
				average_r[radius_r]+=norm[volume_index(norm_CBCT->dim,ni,nj,nk)];
				pixels_r[radius_r]++;
			}
		}
}

average/=(float)pixels;
printf("average(radius %d-%d mm)=%f\n",(int)norm_radius-20,(int)norm_radius,average);

for(ni=0; ni<norm_CBCT->dim[0]; ni++)
		average_r[ni]/=pixels_r[ni];
for(ni=0; ni<=2; ni++)
	average_r[ni]=average_r[3];

LowPass(140,average_r);

for (nj=0; nj<norm_CBCT->dim[1]; nj++){
	for (ni=0; ni<norm_CBCT->dim[0]; ni++){
		float x=(float)ni*norm_CBCT->pix_spacing[0]+norm_CBCT->offset[0]-norm_CBCT->pix_spacing[0];
		float y=(float)nj*norm_CBCT->pix_spacing[1]+norm_CBCT->offset[1]-norm_CBCT->pix_spacing[1];
		radius=sqrt(x*x+y*y);
		radius_r=(int)(radius/norm_CBCT->pix_spacing[0]+0.5);
		for (nk=0; nk<norm_CBCT->dim[2]; nk++){
			if (radius>norm_radius)
				norm[volume_index(norm_CBCT->dim,ni,nj,nk)]=0.0f;
			else{
				norm[volume_index(norm_CBCT->dim,ni,nj,nk)]=average-average_r[radius_r];
			}
		}
	}
}
		printf("The norm mha is processed\n");
		free(average_r);
		free(pixels_r);
}

void bowtie_correction(Volume * vol,MGHCBCT_Options_ext* options)
{
 Volume * norm_CBCT;
 float * img, * norm;
 int ni,nj,nk;
 int i,j,k;

 	if (options->full_fan){
		norm_CBCT=read_mha(options->Full_normCBCT_name);
	}
	else{
		norm_CBCT=read_mha(options->Half_normCBCT_name);
	}
 //norm_CBCT=volume_clone(vol);
 process_norm_CBCT(norm_CBCT,options);
 img=(float *)vol->img;
 norm=(float *)norm_CBCT->img;

 //norm_CBCT=read_mha("C:/AAAfiles/CatPhantom_Full_Fan_Half_Scan/Scan0/Full_norm.mha");

 for (k=0; k<vol->dim[2]; k++){ 
	 nk=(int)floor((k*vol->pix_spacing[2]+vol->offset[2]-vol->pix_spacing[2]-(norm_CBCT->offset[2]-norm_CBCT->pix_spacing[2]))/norm_CBCT->pix_spacing[2]);
	 if ((nk<0)||(nk>=norm_CBCT->dim[2]))
		 continue;
	 for (j=0; j<vol->dim[1]; j++){
		 nj=(int)floor((j*vol->pix_spacing[1]+vol->offset[1]-vol->pix_spacing[1]-(norm_CBCT->offset[1]-norm_CBCT->pix_spacing[1]))/norm_CBCT->pix_spacing[1]);
		 if ((nj<0)||(nj>=norm_CBCT->dim[1]))
			 continue;
		 for (i=0; i<vol->dim[0]; i++){
			 ni=(int)floor((i*vol->pix_spacing[0]+vol->offset[0]-vol->pix_spacing[0]-(norm_CBCT->offset[0]-norm_CBCT->pix_spacing[0]))/norm_CBCT->pix_spacing[0]);	
			 if ((ni<0)||(ni>=norm_CBCT->dim[0]))
				 continue;
			 img[volume_index (vol->dim, i,j,k)]+=norm[volume_index(norm_CBCT->dim,ni,nj,nk)];
		 }
	 }
 }
 free(norm_CBCT->img);
 free(norm_CBCT);
}


//}