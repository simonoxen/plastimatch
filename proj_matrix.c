/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mathutil.h"
#include "volume.h"
#include "MGHMtx_opts.h"
#include "proj_matrix.h"
void set_image_parms (MGHMtx_Options * options);

#define DRR_PLANE_RAY_TOLERANCE 1e-8
#define DRR_STRIDE_TOLERANCE 1e-10
#define DRR_HUGE_DOUBLE 1e10
#define DRR_LEN_TOLERANCE 1e-6
#define DRR_TOPLANE_TOLERANCE 1e-7

#define MSD_NUM_BINS 60
#define LINELEN 128

// #define ULTRA_VERBOSE 1
// #define VERBOSE 1

#define PREPROCESS_ATTENUATION 0
#define IMGTYPE float

#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif
#ifndef M_TWOPI
#define M_TWOPI         (M_PI * 2.0)
#endif

void
proj_matrix_write (double* cam, 
		   double* tgt, double* vup,
		   double sid, double* ic,
		   double* ps, int* ires,
		   int varian_mode, 
		   char* out_fn)
{
    double extrinsic[16];
    double intrinsic[12];
    double projection[12];
    const int cols = 4;
    double sad;

    double nrm[3];
    double vrt[3];
    double vup_tmp[3];  /* Don't overwrite vup */

    FILE* fp;

    vec_zero (extrinsic, 16);
    vec_zero (intrinsic, 12);

    /* Compute image coordinate sys (nrm,vup,vrt) relative to room coords.
       ---------------
       nrm = tgt - cam
       vrt = nrm x vup
       vup = vrt x nrm
       ---------------
    */
    vec3_sub3 (nrm, tgt, cam);
    vec3_normalize1 (nrm);
    vec3_cross (vrt, nrm, vup);
    vec3_normalize1 (vrt);
    vec3_cross (vup_tmp, vrt, nrm);
    vec3_normalize1 (vup_tmp);

    /* !!! But change nrm here to -nrm */
    vec3_scale2 (nrm, -1.0);

    /* Build extrinsic matrix */
    if (varian_mode) {
	vec3_scale2 (vrt, -1.0);
	vec3_copy (&extrinsic[0], nrm);
	vec3_copy (&extrinsic[4], vup_tmp);
	vec3_copy (&extrinsic[8], vrt);
    } else {
	vec3_copy (&extrinsic[0], vrt);
	vec3_copy (&extrinsic[4], vup_tmp);
	vec3_copy (&extrinsic[8], nrm);
    }

    sad = vec3_len (cam);
    m_idx(extrinsic,cols,2,3) = - sad;
    m_idx(extrinsic,cols,3,3) = 1.0;

    /* Build intrinsic matrix */

    m_idx(intrinsic,cols,0,1) = - 1 / ps[0];
    m_idx(intrinsic,cols,1,0) = 1 / ps[1];
    m_idx(intrinsic,cols,2,2) = - 1 / sid;
    //    m_idx(intrinsic,cols,0,3) = ic[0];
    //    m_idx(intrinsic,cols,1,3) = ic[1];

    mat_mult_mat (projection, intrinsic,3,4, extrinsic,4,4);

#if defined (VERBOSE)
    printf ("Extrinsic:\n");
    matrix_print_eol (stdout, extrinsic, 4, 4);
    printf ("Intrinsic:\n");
    matrix_print_eol (stdout, intrinsic, 3, 4);
    printf ("Projection:\n");
    matrix_print_eol (stdout, projection, 3, 4);
#endif

    fp = fopen (out_fn, "w");
    if (!fp) {
	fprintf (stderr, "Error opening %s for write\n", out_fn);
	exit (-1);
    }
    fprintf (fp, "%18.8e %18.8e\n", ic[0], ic[1]);
    fprintf (fp,
	     "%18.8e %18.8e %18.8e %18.8e\n" 
	     "%18.8e %18.8e %18.8e %18.8e\n" 
	     "%18.8e %18.8e %18.8e %18.8e\n", 
	     projection[0], projection[1], projection[2], projection[3],
	     projection[4], projection[5], projection[6], projection[7],
	     projection[8], projection[9], projection[10], projection[11]
	    );
    fprintf (fp, "%18.8e\n%18.8e\n", sad, sid);
    fprintf (fp, "%18.8e %18.8e %18.8e\n", nrm[0], nrm[1], nrm[2]);
    fprintf (fp,
	     "Extrinsic\n"
	     "%18.8e %18.8e %18.8e %18.8e\n" 
	     "%18.8e %18.8e %18.8e %18.8e\n" 
	     "%18.8e %18.8e %18.8e %18.8e\n"
	     "%18.8e %18.8e %18.8e %18.8e\n", 
	     extrinsic[0], extrinsic[1], extrinsic[2], extrinsic[3],
	     extrinsic[4], extrinsic[5], extrinsic[6], extrinsic[7],
	     extrinsic[8], extrinsic[9], extrinsic[10], extrinsic[11],
	     extrinsic[12], extrinsic[13], extrinsic[14], extrinsic[15]
	    );
    fprintf (fp,
	     "Intrinsic\n"
	     "%18.8e %18.8e %18.8e %18.8e\n" 
	     "%18.8e %18.8e %18.8e %18.8e\n" 
	     "%18.8e %18.8e %18.8e %18.8e\n", 
	     intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3],
	     intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7],
	     intrinsic[8], intrinsic[9], intrinsic[10], intrinsic[11]
	    );
    fclose (fp);
}

#if defined (commentout)
void
set_isocenter (Volume* vol, MGHMtx_Options* options)
{
    vol->xmin += options->isocenter[0];
    vol->xmax += options->isocenter[0];
    vol->ymin += options->isocenter[1];
    vol->ymax += options->isocenter[1];
    vol->zmin += options->isocenter[2];
    vol->zmax += options->isocenter[2];
}
#endif

int
read_ProjAngle(char *ProjAngle_file, float *ProjAngle)
{

	FILE *fp;
	char linebuf[LINELEN];
	int nProj=0;
    fp = fopen (ProjAngle_file,"rb");
    if (!fp) {
	fprintf (stderr, "File %s not found\n", ProjAngle_file);
	return 0;
    }
	while (fgets(linebuf,LINELEN,fp)) {
		sscanf (linebuf, "%f",&ProjAngle[nProj++]);
	}
	fclose(fp);
	return(nProj);
}

void
proj_matrix_write_varian_dir (MGHMtx_Options* options)
{
    int a;

    //    double cam_ap[3] = {0.0, -1.0, 0.0};
    //    double cam_lat[3] = {-1.0, 0.0, 0.0};
    //    double* cam = cam_ap;
    //    double* cam = cam_lat;

    double vup[3] = {0, 0, 1};
    double tgt[3] = {0.0, 0.0, 0.0};
    double nrm[3];
    double tmp[3];
    float ProjAngle[1000];
    int varian_mode = 1;

    /* Set source-to-axis distance */
    double sad = options->sad;

    /* Set source-to-image distance */
    double sid = options->sid;

    /* Set image resolution */
    int ires[2] = { options->image_resolution[0],
		    options->image_resolution[1] };

    /* Set physical size of imager in mm */
    //    int isize[2] = { 300, 400 };      /* Actual resolution */
    int isize[2] = { options->image_size[0],
		     options->image_size[1] };

    /* Set ic = image center (in pixels), and ps = pixel size (in mm)
       Note: pixels are numbered from 0 to ires-1 */
    double ic[2] = { options->image_center[0],
		     options->image_center[1] };

    /* Set pixel size in mm */
    double ps[2] = { (double)isize[0]/(double)ires[0], 
		     (double)isize[1]/(double)ires[1] };

    /* Loop through camera angles */
    //options->angle_diff=30.0f;
    int nProj=read_ProjAngle(options->ProjAngle_file,ProjAngle);

    //for (a = 0; a < options->num_angles; a++) {
    for (a = 0; a < nProj; a++) {
	double angle = ProjAngle[a];
	double cam[3];
	char out_fn[256];
	//char multispectral_fn[256];
	//angle=0;

	cam[0] = cos(angle/180.0*3.14159);
	cam[1] = sin(angle/180.0*3.14159);
	cam[2] = 0.0;

	//printf ("Rendering DRR %d\n", a);

	/* Place camera at distance "sad" from the volume isocenter */
	vec3_sub3 (nrm, tgt, cam);
	vec3_normalize1 (nrm);
	vec3_scale3 (tmp, nrm, sad);
	vec3_copy (cam, tgt);
	vec3_sub2 (cam, tmp);

	/* Some debugging info */
#if defined (VERBOSE)
	vec_set_fmt ("%12.4g ");
	printf ("cam: ");
	vec3_print_eol (stdout, cam);
	printf ("tgt: ");
	vec3_print_eol (stdout, tgt);
	printf ("ic:  %g %g\n", ic[0], ic[1]);
#endif
	sprintf (out_fn, "%s%04d.txt", options->output_prefix, a);
	proj_matrix_write (cam, tgt, vup, 
			   sid, ic, ps, ires, 
			   varian_mode, out_fn);
    }
}

//	int main(int argc, char* argv[])
//{
//	MGHMtx_Options options;
//
//	parse_args (&options, argc, argv);
//
////#if defined (PREPROCESS_ATTENUATION)
////	preprocess_attenuation (vol);
////#endif
//	drr_render_volumes (&options);
//
//	return 0;
//}

void write_matrix(MGHMtx_Options * options)
{

//#if defined (PREPROCESS_ATTENUATION)
//	preprocess_attenuation (vol);
	set_image_parms (options);
//#endif
	//drr_render_volumes (options);
	proj_matrix_write_varian_dir (options);
}


