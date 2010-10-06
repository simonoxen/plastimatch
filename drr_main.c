/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "drr.h"
#include "drr_cuda.h"
#include "drr_opts.h"
#include "drr_trilin.h"
#include "math_util.h"
#include "mha_io.h"
#include "plm_timer.h"
#include "proj_image.h"
#include "proj_matrix.h"
#include "delayload.h"

static void*
allocate_gpu_memory (
    Proj_image *proj, 
    Volume *vol,
    Drr_options *options
)
{
#if CUDA_FOUND
    switch (options->threading) {
    case THREADING_BROOK:
    case THREADING_CUDA:
    cudaDetect ();
    case THREADING_OPENCL:
	return drr_cuda_state_create (proj, vol, options);
    case THREADING_CPU_SINGLE:
    case THREADING_CPU_OPENMP:
    default:
	return 0;
    }
#else
    return 0;
#endif
}

static void
create_matrix_and_drr (
    Volume* vol, 
    Proj_image *proj,
    double cam[3],
    double tgt[3],
    double nrm[3],
    int a, 
    void *dev_state, 
    Drr_options* options
)
{
    char mat_fn[256];
    char img_fn[256];
    char multispectral_fn[256];
    Proj_matrix *pmat = proj->pmat;
    double vup[3] = {
	options->vup[0],
	options->vup[1],
	options->vup[2] };
    double sid = options->sid;
    Timer timer;

    /* Set ic = image center (in pixels), and ps = pixel size (in mm)
       Note: pixels are numbered from 0 to ires-1 */
    double ic[2] = { options->image_center[0],
		     options->image_center[1] };

    /* Set image resolution */
    int ires[2] = { options->image_resolution[0],
		    options->image_resolution[1] };

    /* Set physical size of imager in mm */
    int isize[2] = { options->image_size[0],
		     options->image_size[1] };

    /* Set pixel size in mm */
    double ps[2] = { (double)isize[0]/(double)ires[0], 
		     (double)isize[1]/(double)ires[1] };

    /* Create projection matrix */
    sprintf (mat_fn, "%s%04d.txt", options->output_prefix, a);
    proj_matrix_set (pmat, cam, tgt, vup, sid, ic, ps, ires);

    if (options->output_format == OUTPUT_FORMAT_PFM) {
	sprintf (img_fn, "%s%04d.pfm", options->output_prefix, a);
    } else if (options->output_format == OUTPUT_FORMAT_PGM) {
	sprintf (img_fn, "%s%04d.pgm", options->output_prefix, a);
    } else {
	sprintf (img_fn, "%s%04d.raw", options->output_prefix, a);
    }
    sprintf (multispectral_fn, "%s%04d.msd", options->output_prefix, a);

    drr_render_volume_perspective (proj, vol, ps, dev_state, options);

    plm_timer_start (&timer);
    proj_image_save (proj, img_fn, mat_fn);
    printf ("I/O time: %f sec\n", plm_timer_report (&timer));
}

/* All distances in mm */
void
drr_render_volume (Volume* vol, Drr_options* options)
{
    Proj_image *proj;
    Proj_matrix *pmat;
    int a;
    Timer timer;
    void *dev_state = 0;

    /* tgt is isocenter */
    double tgt[3] = {
	options->isocenter[0],
	options->isocenter[1],
	options->isocenter[2] };

    plm_timer_start (&timer);

    /* Allocate data for image and matrix */
    proj = proj_image_create ();
    proj_image_create_pmat (proj);
    proj_image_create_img (proj, options->image_resolution);
    pmat = proj->pmat;

    /* Allocate memory on the gpu device */
    dev_state = allocate_gpu_memory (proj, vol, options);

    /* If nrm was specified, only create a single image */
    if (options->have_nrm) {
	double cam[3];
	double nrm[3] = {
	    options->nrm[0],
	    options->nrm[1],
	    options->nrm[2] };

	/* Make sure nrm is normal */
	vec3_normalize1 (nrm);

	/* Place camera at distance "sad" from the volume isocenter */
	cam[0] = tgt[0] + options->sad * nrm[0];
	cam[1] = tgt[1] + options->sad * nrm[1];
	cam[2] = tgt[2] + options->sad * nrm[2];

	create_matrix_and_drr (vol, proj, cam, tgt, nrm, 0, 
	    dev_state, options);
    }

    /* Otherwise, loop through camera angles */
    else {
	for (a = 0; a < options->num_angles; a++) {
	    double angle = a * options->angle_diff;
	    double cam[3];
	    double nrm[3];

	    printf ("Rendering DRR %d\n", a);

	    /* Place camera at distance "sad" from the volume isocenter */
	    cam[0] = tgt[0] + options->sad * cos(angle);
	    cam[1] = tgt[1] - options->sad * sin(angle);
	    cam[2] = tgt[2];
	
	    /* Compute normal vector */
	    vec3_sub3 (nrm, tgt, cam);
	    vec3_normalize1 (nrm);

	    create_matrix_and_drr (vol, proj, cam, tgt, nrm, a, 
		dev_state, options);
	}
    }
    proj_image_destroy (proj);

#if CUDA_FOUND
    if (dev_state) {
	drr_cuda_state_destroy (dev_state);
    }
#endif

    printf ("Total time: %g secs\n", plm_timer_report (&timer));
}

void
set_isocenter (Volume* vol, Drr_options* options)
{
    vol->offset[0] -= options->isocenter[0];
    vol->offset[1] -= options->isocenter[1];
    vol->offset[2] -= options->isocenter[2];
}

int
main (int argc, char* argv[])
{
    Volume* vol;
    Drr_options options;

    parse_args (&options, argc, argv);

    vol = read_mha (options.input_file);
    if (!vol) return -1;

    volume_convert_to_float (vol);

    switch (options.threading) {
#if OPENCL_FOUND
    case THREADING_OPENCL:
	preprocess_attenuation_and_drr_render_volume_cl (vol, &options);
	break;
#endif
    default:
#if defined (DRR_PREPROCESS_ATTENUATION)
	drr_preprocess_attenuation (vol);
#endif
	drr_render_volume (vol, &options);
	break;
    }

    volume_destroy (vol);
    printf ("Done.\n");
    return 0;
}
