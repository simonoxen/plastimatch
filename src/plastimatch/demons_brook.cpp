/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <brook/brook.hpp>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "convolution_brook_kernel.h"
#include "demons_brook_kernel.h"
#include "demons_opts.h"
#include "demons_misc.h"
#include "mha_io.h"
#include "plm_timer.h"
#include "volume.h"
#include "vf_stats.h"

/* Function computes the image gradient in the x, y, and z directions for the static image */
float**
estimate_partial_derivatives (Volume* vol1)
{
	int i;
	float** der;
	short* temp1 = (short*)vol1->img;
	float3 dim;

	// Obtain the texture size in the X and Y dimensions needed to store the volume
	int size = (int)ceil(sqrt((double)vol1->npix/4));
	printf("A texture memory of %d x %d is needed to accommodate the volume \n", size, size);

	int texture_size = size*size; // Texture of float4 elements
	float* img1 = (float*)malloc(sizeof(float)*texture_size*4);
	if(img1 == NULL){
		printf("Memory allocaton failed.....Exiting\n");
		exit(1);
	}	
	for(i=0; i < texture_size; i++) // Initialize the stream
		img1[i] = 0.0;
	
	dim.x = (float)vol1->dim[0];
	dim.y = (float)vol1->dim[1];
	dim.z = (float)vol1->dim[2];
	
	for(i=0; i < vol1->npix; i++) // Read the image
		img1[i] = (float)temp1[i];

	printf("Estimating partial derivatives \n");
	
	der = (float**)malloc(3*sizeof(float*));
	if(!der){
		printf("Memory allocation failed for stage 1..........Exiting\n");
		exit(1);
	}
	for(i=0; i < 3; i++){
		der[i] = (float*)malloc(sizeof(float)*texture_size*4);
		if(!der[i]){
			printf("Memory allocation failed for stage 2, dimension %d..........Exiting\n",i);
			exit(1);
		}
	}

	Timer timer;
	plm_timer_start (&timer);

	/* Compute partial derivatives on the GPU */
	::brook::stream vol_st1(::brook::getStreamType(( float4  *)0), size , size, -1);
	::brook::stream der_st(::brook::getStreamType(( float4 *)0), size , size,-1);

	streamRead(vol_st1, img1);
	
	k_epdx(vol_st1, (float)size, dim, vol1->pix_spacing[0],der_st);
	streamWrite(der_st, der[0]);

	k_epdy(vol_st1, (float)size, dim, vol1->pix_spacing[1], der_st);
	streamWrite(der_st, der[1]);

	k_epdz(vol_st1, (float)size, dim, vol1->pix_spacing[2], der_st);
	streamWrite(der_st, der[2]);

	printf ("Compute image intensity gradient on the GPU = %f secs\n", 
		plm_timer_report (&timer));

	free(img1);
	return (der);
}

/* Vector fields are all in mm units */
Volume*
demons_brook_internal (Volume* fixed, Volume* moving, Volume* moving_grad, 
		       Volume* vf_init, DEMONS_Parms* parms)
{
    int it;
    float3 m_offset, f_offset;              /* Copied from input volume */
    float3 m_pix_spacing, f_pix_spacing;    /* Copied from input volume */
    float3 f_dim, m_dim;		    /* Copied from input volume */
    int f_size, m_size;			    /* Textures are size x size pixels */
    float* f_img = (float*) fixed->img;     /* Raw pointer to data */
    float* m_img = (float*) moving->img;    /* Raw pointer to data */
    float **vf_img, **mg_img;		    /* Raw pointers to data */
    float *kerx, *kery, *kerz;		    /* Gaussian filters */
    int fw[3];				    /* Filter widths */
    Volume* vf;				    /* Output vector field */
    Volume* debug_vol;			    /* Only used when debugging */
    double io_cycles = 0;		    /* Timing for I/O */
    double processing_cycles = 0;	    /* Timing for GPU processing */

    /* Compute texture sizes.  Textures are float4, thus divide by 4. */
    f_size = (int) ceil (sqrt((double)fixed->npix/4));
    m_size = (int) ceil (sqrt((double)moving->npix/4));
    int f_tex_size = f_size * f_size;
    int m_tex_size = m_size * m_size;

    printf ("Texture size is %d x %d for fixed image volumes (%d pix)\n", f_size, f_size, fixed->npix);
    printf ("Texture size is %d x %d for moving image volumes (%d pix)\n", m_size, m_size, moving->npix);

    /* Allocate memory for temporary images needed for data exchange with brook. */
    float* f_img_tex = (float*) malloc (sizeof(float)*f_tex_size*4);
    if (!f_img_tex) {
	printf("Couldn't allocate memory for temporary image.\n");
	exit(-1);
    }
    memset (f_img_tex, 0, sizeof(float)*f_tex_size*4);
    memcpy (f_img_tex, f_img, fixed->npix * sizeof(float));

    /* GCS FIX: Need to compute different size for moving image */
    float* m_img_tex = (float*) malloc (sizeof(float)*m_tex_size*4);
    if (!m_img_tex) {
	printf("Couldn't allocate memory for temporary image.\n");
	exit(-1);
    }
    memset (m_img_tex, 0, sizeof(float)*m_tex_size*4);
    memcpy (m_img_tex, m_img, moving->npix * sizeof(float));

    /* Initialize the moving image gradient */
    vf_convert_to_planar (moving_grad, m_tex_size*4);
    mg_img = (float**) moving_grad->img;

    /* Initialize various fixed inputs for brook kernel */
    f_dim.x = (float) fixed->dim[0];
    f_dim.y = (float) fixed->dim[1];
    f_dim.z = (float) fixed->dim[2];
    m_dim.x = (float) moving->dim[0];
    m_dim.y = (float) moving->dim[1];
    m_dim.z = (float) moving->dim[2];
    f_offset.x = fixed->offset[0];
    f_offset.y = fixed->offset[1];
    f_offset.z = fixed->offset[2];
    m_offset.x = moving->offset[0];
    m_offset.y = moving->offset[1];
    m_offset.z = moving->offset[2];
    f_pix_spacing.x = fixed->pix_spacing[0];
    f_pix_spacing.y = fixed->pix_spacing[1];
    f_pix_spacing.z = fixed->pix_spacing[2];
    m_pix_spacing.x = moving->pix_spacing[0];
    m_pix_spacing.y = moving->pix_spacing[1];
    m_pix_spacing.z = moving->pix_spacing[2];

    /* Validate filter widths */
    validate_filter_widths (fw, parms->filter_width);

    /* Create the seperable smoothing kernels for the x, y, and z directions */
    kerx = create_ker (parms->filter_std / fixed->pix_spacing[0], fw[0]/2);
    kery = create_ker (parms->filter_std / fixed->pix_spacing[1], fw[1]/2);
    kerz = create_ker (parms->filter_std / fixed->pix_spacing[2], fw[2]/2);
    kernel_stats (kerx, kery, kerz, fw);

    /* Initial guess for displacement field */
    if (vf_init) {
	vf = volume_clone (vf_init);
	vf_convert_to_planar (vf, f_tex_size*4);
    } else {
	vf = volume_create (fixed->dim, fixed->offset, fixed->pix_spacing, PT_VF_FLOAT_PLANAR, fixed->direction_cosines, f_tex_size*4);
    }

    /* Allocate the debug volume */
    debug_vol = volume_create (fixed->dim, fixed->offset, fixed->pix_spacing, PT_FLOAT, fixed->direction_cosines, 0);
    ::brook::stream s_debug(::brook::getStreamType(( float4  *)0), f_size , f_size, -1);
 
    /* Allocate memory for the various streams.  At the beginning and end 
       of each loop, the current estimate is in s_vf_smooth_x, s_vf_smooth_y and s_vf_smooth_z.  */
    ::brook::stream s_vf_smooth_x(::brook::getStreamType(( float4  *)0), f_size , f_size, -1);
    ::brook::stream s_vf_smooth_y(::brook::getStreamType(( float4  *)0), f_size , f_size, -1);
    ::brook::stream s_vf_smooth_z(::brook::getStreamType(( float4  *)0), f_size , f_size, -1);
    ::brook::stream st_vf_temp_x(::brook::getStreamType(( float4  *)0), f_size , f_size,-1);
    ::brook::stream st_vf_temp_y(::brook::getStreamType(( float4  *)0), f_size , f_size,-1);
    ::brook::stream st_vf_temp_z(::brook::getStreamType(( float4  *)0), f_size , f_size,-1);
    ::brook::stream der_st_x(::brook::getStreamType(( float4  *)0), m_size , m_size, -1);
    ::brook::stream der_st_y(::brook::getStreamType(( float4  *)0), m_size , m_size, -1);
    ::brook::stream der_st_z(::brook::getStreamType(( float4  *)0), m_size , m_size, -1);
    ::brook::stream img_st_1(::brook::getStreamType(( float4  *)0), f_size , f_size, -1);
    ::brook::stream img_st_2(::brook::getStreamType(( float4  *)0), m_size , m_size, -1);

    ::brook::stream s_ssd(::brook::getStreamType(( float4  *)0), 1 , 1, -1);	// Single float4
    ::brook::stream st_kerx(::brook::getStreamType(( float *)0), 1, fw[0], -1); // Five element kernel
    ::brook::stream st_kery(::brook::getStreamType(( float *)0), 1, fw[1], -1); // Five element kernel
    ::brook::stream st_kerz(::brook::getStreamType(( float *)0), 1, fw[2], -1); // Five element kernel

    /* Uncomment this code for display of SSD each iteration (below) */
    ::brook::stream st_vol_diff(::brook::getStreamType(( float4  *)0), f_size , f_size, -1);

    /* Initialize the vector streams with zeros */
    k_initial_vectors4(st_vf_temp_x);
    k_initial_vectors4(st_vf_temp_y);
    k_initial_vectors4(st_vf_temp_z);

    /* Initialize input vf stream to initial guess */
    vf_img = (float**) vf->img;
    streamRead (s_vf_smooth_x, vf_img[0]);
    streamRead (s_vf_smooth_y, vf_img[1]);
    streamRead (s_vf_smooth_z, vf_img[2]);

    Timer timer;
    plm_timer_start (&timer);

    /* Read data into the stream structures used by Brook */
    streamRead (der_st_x, mg_img[0]);
    streamRead (der_st_y, mg_img[1]);
    streamRead (der_st_z, mg_img[2]);
    streamRead (img_st_1, f_img_tex);
    streamRead (img_st_2, m_img_tex);
    streamRead (st_kerx, kerx);
    streamRead (st_kery, kery);
    streamRead (st_kerz, kerz);

    io_cycles += plm_timer_report (&timer);

    plm_timer_start (&timer);

    for (it = 0; it < parms->max_its; it++) {

	// Update x component of the displacement field 
	printf (".");
	k_evf_gcs (
	    s_vf_smooth_x,	    // input stream
	    s_vf_smooth_x,	    // x displacement in voxels
	    s_vf_smooth_y,	    // y displacement in voxels
	    s_vf_smooth_z,	    // z displacement in voxels
	    der_st_x,	    // nabla_x
	    der_st_x,	    // nabla_x
	    der_st_y,	    // nabla_y
	    der_st_z,	    // nabla_z
	    img_st_1,	    // static image
	    img_st_2,	    // moving image
	    parms->homog,   // homogeneity factor 
	    parms->accel,   // acceleration
	    parms->denominator_eps,  // denominator epsilon
	    (float) f_size,    // texture memory dimensions
	    (float) m_size,    // texture memory dimensions
	    f_dim,	    // volume dimensions
	    m_dim,	    // volume dimensions
	    f_offset,	    // fixed image origin
	    m_offset,	    // moving image origin
	    f_pix_spacing,  // fixed image voxel size
	    m_pix_spacing,  // moving image voxel size
	    st_vf_temp_x);  // output stream

	// Update y component of the displacement field 
	k_evf_gcs (
	    s_vf_smooth_y,	    // input stream
	    s_vf_smooth_x,	    // x displacement in voxels
	    s_vf_smooth_y,	    // y displacement in voxels
	    s_vf_smooth_z,	    // z displacement in voxels
	    der_st_y,	    // nabla_y
	    der_st_x,	    // nabla_x
	    der_st_y,	    // nabla_y
	    der_st_z,	    // nabla_z
	    img_st_1,	    // static image
	    img_st_2,	    // moving image
	    parms->homog,   // homogeneity factor 
	    parms->accel,   // accleration
	    parms->denominator_eps,  // denominator epsilon
	    (float) f_size,    // texture memory dimensions
	    (float) m_size,    // texture memory dimensions
	    f_dim,	    // volume dimensions
	    m_dim,	    // volume dimensions
	    f_offset,	    // fixed image origin
	    m_offset,	    // moving image origin
	    f_pix_spacing,  // fixed image voxel size
	    m_pix_spacing,  // moving image voxel size
	    st_vf_temp_y);  // output stream

	// Update z component of the displacement field 
	k_evf_gcs (
	    s_vf_smooth_z,	    // input stream
	    s_vf_smooth_x,	    // x displacement in voxels
	    s_vf_smooth_y,	    // y displacement in voxels
	    s_vf_smooth_z,	    // z displacement in voxels
	    der_st_z,	    // nabla_z
	    der_st_x,	    // nabla_x
	    der_st_y,	    // nabla_y
	    der_st_z,	    // nabla_z
	    img_st_1,	    // static image
	    img_st_2,	    // moving image
	    parms->homog,   // homogeneity factor 
	    parms->accel,   // accleration
	    parms->denominator_eps,  // denominator epsilon
	    (float) f_size,    // texture memory dimensions
	    (float) m_size,    // texture memory dimensions
	    f_dim,	    // volume dimensions
	    m_dim,	    // volume dimensions
	    f_offset,	    // fixed image origin
	    m_offset,	    // moving image origin
	    f_pix_spacing,  // fixed image voxel size
	    m_pix_spacing,  // moving image voxel size
	    st_vf_temp_z);  // output stream

	/* GCS Wed Dec 26 16:36:41 EST 2007 
	   I haven't been able to get reduce to work on the GPU.  But the image difference 
	   works just fine.  So I will do everything except the summation on the GPU, and 
	   the summation on the CPU.
	   Ref: http://www.gpgpu.org/forums/viewtopic.php?t=5013
	*/
#if defined (commentout)
	/* Uncomment this code to print SSD each iteration */
	k_volume_difference (
	    s_vf_smooth_x, // x displacement in voxels
	    s_vf_smooth_y, // y displacement in voxels
	    s_vf_smooth_z, // z displacement in voxels
	    img_st_1, // static image
	    img_st_2, // moving image
	    f_dim, // volume dimensions
	    (float) f_size, // texture memory dimensions
	    st_vol_diff); // output stream
	streamWrite (st_vol_diff, f_img_tex);	    /* f_img_tex is not used after streamRead */
	int inliers = 0;
	float ssd = 0.0f;
	for (i = 0; i < fixed->npix; i++) {
	    if (f_img_tex[i] >= 0.0) {
		inliers ++;
		ssd += f_img_tex[i];
	    }
	}
	printf ("----- SSD = %.01f (%d/%d)\n", ssd/inliers, inliers, fixed->npix);
#endif

#if defined (commentout)
	/* Uncomment this to get stats for the vector field */
	streamWrite(st_vf_temp_x, vf_img[0]);
	streamWrite(st_vf_temp_y, vf_img[1]);
	streamWrite(st_vf_temp_z, vf_img[2]);
	vf_print_stats (vf);
	write_mha ("vf_est_brook.mha", vf);
#endif

#if defined (commentout)
	/* Uncomment this to get an intermediate snapshot of the vectorfield */
	k_debug (s_vf_smooth_x, // input stream
		 s_vf_smooth_x, // x displacement in voxels
		 s_vf_smooth_y, // y displacement in voxels
		 s_vf_smooth_z, // z displacement in voxels
		 der_st_x, // nabla_z
		 der_st_x, // nabla_x
		 der_st_y, // nabla_y
		 der_st_z, // nabla_z
		 img_st_1, // static image
		 img_st_2, // moving image
		 dim, // volume dimensions
		 (float)size, // texture memory dimensions
		 s_debug); // output stream
	streamWrite (s_debug, debug_vol->img);
	write_mha ("debug.mha", debug_vol);
#endif

	// Smooth the displacement vectors along the x direction
	k_conv_x (st_vf_temp_x, st_kerx, fw[0], f_dim, (float) f_size, s_vf_smooth_x);
	k_conv_x (st_vf_temp_y, st_kerx, fw[0], f_dim, (float) f_size, s_vf_smooth_y);
	k_conv_x (st_vf_temp_z, st_kerx, fw[0], f_dim, (float) f_size, s_vf_smooth_z);

	// Smooth displacement vectors along the y direction
	k_conv_y (s_vf_smooth_x, st_kery, fw[1], f_dim, (float) f_size, st_vf_temp_x);
	k_conv_y (s_vf_smooth_y, st_kery, fw[1], f_dim, (float) f_size, st_vf_temp_y);
	k_conv_y (s_vf_smooth_z, st_kery, fw[1], f_dim, (float) f_size, st_vf_temp_z);

	// Smooth displacement vectors along the z direction  
	k_conv_z (st_vf_temp_x, st_kerz, fw[2], f_dim, (float) f_size, s_vf_smooth_x);
	k_conv_z (st_vf_temp_y, st_kerz, fw[2], f_dim, (float) f_size, s_vf_smooth_y);
	k_conv_z (st_vf_temp_z, st_kerz, fw[2], f_dim, (float) f_size, s_vf_smooth_z);

#if defined (commentout)
	/* Uncomment this to get stats for the vector field */
	streamWrite (s_vf_smooth_x, vf_img[0]);
	streamWrite (s_vf_smooth_y, vf_img[1]);
	streamWrite (s_vf_smooth_z, vf_img[2]);
	vf_print_stats (vf);
	write_mha ("vf_smooth_brook.mha", vf);
#endif
    }

    processing_cycles += plm_timer_report (&timer);

    printf("\n");
	
    plm_timer_start (&timer);

    /* Read back the displacement field in terms of voxels */
    streamWrite(s_vf_smooth_x, vf_img[0]);
    streamWrite(s_vf_smooth_y, vf_img[1]);
    streamWrite(s_vf_smooth_z, vf_img[2]);

    io_cycles += plm_timer_report (&timer);

    /* Read back the displacement field in terms of mm */
    /*
      streamWrite(s_vf_smooth_x, vec_mm[0]);
      streamWrite(s_vf_smooth_y, vec_mm[1]);
      streamWrite(s_vf_smooth_z, vec_mm[2]);
      write_to_file("gpu_vector_field_magnitude.txt", vol1->npix, vec_mm);
    */

    printf ("I/O time = %f\n", io_cycles);
    printf ("Processing time on GPU = %f\n", processing_cycles);
    printf("Total time = %f\n", io_cycles + processing_cycles);

    printf ("Converting to interleaved\n");
    vf_convert_to_interleaved (vf);

    free (m_img_tex);
    free (f_img_tex);
    return vf;
}

extern "C" {
void
conv_test (Volume *vf, float* ker, int ker_size)
{
    float **img;
    float3 dim;
    int size;

    size = (int) ceil (sqrt((double)vf->npix/4)); // Size of the texture to allocate
    printf ("Allocating texture of size %d x %d to store volume information \n", size, size);
    int texture_size = size*size; // Size of the texture needed on the GPU

    dim.x = (float) vf->dim[0];
    dim.y = (float) vf->dim[1];
    dim.z = (float) vf->dim[2];

    printf ("Converting to planar\n");
    vf_convert_to_planar (vf, 0);
    
    printf ("Padding\n");
    vf_pad_planar (vf, texture_size*4*sizeof(float));

    /* Allocate memory for the various streams */
    ::brook::stream st_vf_x(::brook::getStreamType(( float4  *)0), size , size, -1);
    ::brook::stream st_vf_y(::brook::getStreamType(( float4  *)0), size , size, -1);
    ::brook::stream st_vf_z(::brook::getStreamType(( float4  *)0), size , size, -1);
    ::brook::stream st_vf_out(::brook::getStreamType(( float4  *)0), size , size, -1);
    ::brook::stream st_ker(::brook::getStreamType(( float *)0), 1, 5, -1); // Five element kernel

    /* Read data into the stream structures used by Brook */
    printf ("Sending data to brook\n");
    img = (float**) vf->img;
    streamRead (st_vf_x, img[0]);
    streamRead (st_vf_y, img[1]);
    streamRead (st_vf_z, img[2]);
    streamRead (st_ker, ker);

    /* Test out convolution */
    printf ("Testing convolution\n");
    k_conv_x(st_vf_x, st_ker, ker_size, dim, size, st_vf_out);	
    streamWrite (st_vf_out, img[0]);
    k_conv_x(st_vf_y, st_ker, ker_size, dim, size, st_vf_out);	
    streamWrite (st_vf_out, img[1]);
    k_conv_x(st_vf_z, st_ker, ker_size, dim, size, st_vf_out);	
    streamWrite (st_vf_out, img[2]);

    printf ("Converting to interleaved\n");
    vf_convert_to_interleaved (vf);
}
}

void
convolution_test (void)
{
    Volume *vf1, *vf2;
    int dim[] = { 128, 128, 128 };
    float offset[] = { 0.0, 0.0, 0.0 };
    float pix_spacing[] = { 1.0, 1.0, 1.0 };
    int i, j, k, v;
    float* img;
    float ker[5] = { 0.1f, 0.2f, 0.4f, 0.2f, 0.1f };

    vf1 = volume_create (dim, offset, pix_spacing, PT_VF_FLOAT_INTERLEAVED, 0, 0);
    img = (float*) vf1->img;
    printf ("Building random numbers...\n");
    for (k = 0, v = 0; k < dim[2]; k++) {
	for (j = 0; j < dim[1]; j++) {
	    for (i = 0; i < dim[0]; i++, v++) {
		float* xyz = &img[3*v];
		xyz[0] = ((float) rand()) / RAND_MAX;
		xyz[1] = ((float) rand()) / RAND_MAX;
		xyz[2] = ((float) rand()) / RAND_MAX;
	    }
	}
    }
    printf ("Cloning image\n");
    vf2 = volume_clone (vf1);

    write_mha ("vf_ori.mha", vf1);
    vf_print_stats (vf1);

    printf ("Convolving on cpu\n");
    vf_convolve_x (vf2, vf1, ker, 5);
    vf_print_stats (vf2);
    write_mha ("vf_c.mha", vf2);

    printf ("Convolving on gpu\n");
    conv_test (vf1, ker, 5);
    vf_print_stats (vf1);

    write_mha ("vf_brook.mha", vf1);

    printf ("Done\n");
}

extern "C" {
Volume*
demons_brook (Volume* fixed, Volume* moving, Volume* moving_grad, 
	      Volume* vf_init, DEMONS_Parms* parms)
{
    Volume* vf;
    Timer timer;
    double diff_run;

    /* Run the algorithm */
    plm_timer_start (&timer);
    vf = demons_brook_internal (fixed, moving, moving_grad, vf_init, parms);
    diff_run = plm_timer_report (&timer);
    printf("Time for %d iterations = %f\n", parms->max_its, diff_run);
    printf ("Done!\n");

    return vf;
}
} /* extern "C" */
