/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "bowtie_correction.h"
#include "fdk.h"
#include "fdk_brook.h"
#include "fdk_cuda.h"
#include "fdk_opencl.h"
#include "fdk_opts.h"
#include "fdk_util.h"
#include "file_util.h"
#include "math_util.h"
#include "mha_io.h"
#include "print_and_exit.h"
#include "proj_image.h"
#include "proj_image_dir.h"
#include "plm_timer.h"
#include "delayload.h"

int 
main (int argc, char* argv[])
{
    Fdk_options options;
    Volume* vol;
    Proj_image_dir *proj_dir;
    
    /* Parse command line arguments */
    fdk_parse_args (&options, argc, argv);

    /* Look for input files */
    proj_dir = proj_image_dir_create (options.input_dir);
    if (!proj_dir) {
	print_and_exit ("Error: couldn't find input files in directory %s\n",
	    options.input_dir);
    }

    /* Choose subset of input files if requested */
    if (options.image_range_requested) {
	proj_image_dir_select (proj_dir, options.first_img, 
	    options.skip_img, options.last_img);
    }

    /* Allocate memory */
    vol = my_create_volume (&options);

    printf ("Reconstructing...\n");
    switch (options.threading) {
#if (BROOK_FOUND)
    case THREADING_BROOK:
	fdk_brook_reconstruct (vol, proj_dir, &options);
	break;
#endif
#if (CUDA_FOUND)
    case THREADING_CUDA:
	if (!delayload_cuda()) {
	    // If we continue to attempt to use the CUDA runtime
	    // after failing to load the CUDA runtime, we crash.
	    exit (0);
	}
	CUDA_reconstruct_conebeam (vol, proj_dir, &options);
	break;
#endif
#if (OPENCL_FOUND)
    case THREADING_OPENCL:
	delayload_opencl ();
	//OPENCL_reconstruct_conebeam_and_convert_to_hu (vol, proj_dir, &options);
	opencl_reconstruct_conebeam (vol, proj_dir, &options);
	break;
#endif
    case THREADING_CPU_SINGLE:
    case THREADING_CPU_OPENMP:
    default:
	reconstruct_conebeam (vol, proj_dir, &options);
    }

    /* Free memory */
    proj_image_dir_destroy (proj_dir);

    /* Prepare HU values in output volume */
    convert_to_hu (vol, &options);

    /* Do bowtie filter corrections */
    fdk_do_bowtie (vol, &options);

    /* Write output */
    printf ("Writing output volume(s)...\n");
    write_mha (options.output_file, vol);

    /* Free memory */
    volume_destroy (vol);

    printf(" done.\n\n");

    return 0;
}
