/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>
#include "mha_io.h"
#include "volume.h"

// includes, kernels
#include <demon_cuda_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runDemon( int argc, char** argv);
float** computeNabla(Volume * vol1);
float** estimate_vector_field(Volume* vol1,Volume *vol2,float** der,int n_iter);
Volume* compute_intensity_differences(Volume* vol, Volume* warped);
float* create_ker(float coeff, int kx);
Volume* warp_image(Volume* vol, float** vec);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runDemon( argc, argv);

    //CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Execute Demons algorithm on CUDA
////////////////////////////////////////////////////////////////////////////////
void
runDemon( int argc, char** argv) 
{
#if defined (commentout)
    // declare variables
	Volume* vol1;
    Volume* vol2;
	Volume* diff_before;
    Volume* diff_after;
    Volume* temp;

	char* infile1;
    char* infile2;

	float** der;
	float** vec;

    // create timer
    unsigned int timer;
    cutCreateTimer(&timer);

    // check input arguments
	if(argc != 4){
		printf("Usage: demon_j.exe <filename1> <filename2> n_iter \n");
		printf("filename1 is the static image \n");
		printf("filename2 is the moving image \n");
		exit(1);
	}
    
    // read the image and create first volume
	infile1 = argv[1];
	printf("Reading file: %s \n", infile1);
    vol1 = read_mha(infile1);

    // read the image and create second volume
	infile2 = argv[2];
	printf("Reading file: %s \n", infile2);
    vol2 = read_mha(infile2);

    // read number of iterations as an int
	int n_iter = atoi(argv[3]);

    // make sure the two volumes are same size
	if(vol1->npix != vol2->npix){
		printf("Files are of different sizes.....Exiting\n");
		exit(1);
	}

    // detect and check device
    CUT_CHECK_DEVICE();

    //compute initial difference between volumes
    diff_before = compute_intensity_differences(vol1,vol2);
	write_mha("differences_before_registration.mha",diff_before);

    // compute partial derivative of static volume using gpu
    der = computeNabla(vol1);

    // estimate vector field using gpu
    cutStartTimer(timer);
    vec = estimate_vector_field(vol1,vol2,der,n_iter);
    cutStopTimer(timer);
    printf("\nTime to estimate vector field = %f s \n",cutGetTimerValue(timer)/1000);
    cutDeleteTimer(timer);

    // create and write warped volume
	temp = warp_image(vol2, vec);
	write_mha("warped.mha", temp);
	
	// compute differences between original and warped volume
	diff_after = compute_intensity_differences(vol1, temp);
	write_mha("differences_after_registration.mha",diff_after);

    //free memory
    free(der);
    free(vec);
    free(diff_before);
    free(diff_after);
    free(temp);
    free(vol1);
    free(vol2);
#endif
}


Volume* 
compute_intensity_differences(Volume* vol, Volume* warped)
{
	Volume* temp = (Volume*)malloc(sizeof(Volume));
	if(!temp){
		printf("Memory allocation failed for volume...Exiting\n");
		exit(1);
	}
	
	for(int i=0;i<3; i++){
		temp->dim[i] = vol->dim[i];
		temp->offset[i] = vol->offset[i];
		temp->pix_spacing[i] = vol->pix_spacing[i];
	}

	temp->npix = vol->npix;
	temp->pix_type = vol->pix_type;
	temp->xmax = vol->xmax;
	temp->xmin = vol->xmin;
	temp->ymax = vol->ymax;
	temp->ymin = vol->ymin;
	temp->zmax = vol->zmax;
	temp->zmin = vol->zmin;

	temp->img = (void*)malloc(sizeof(short)*temp->npix);
	memset (temp->img, -1200, sizeof(short)*temp->npix);

	int p = 0; // Voxel index

	short* temp2 = (short*)vol->img;
	short* temp1 = (short*)warped->img;
	short* temp3 = (short*)temp->img;

	for(int i=0; i < vol->dim[2]; i++)
		for(int j=0; j < vol->dim[1]; j++)
			for(int k=0; k < vol->dim[0]; k++){
				temp3[p] = (temp2[p] - temp1[p]) - 1200;
				p++;
			}

	return temp;
}


float* 
create_ker(float coeff, int kx){
	int i,j=0;
	float sum = 0.0;
	int num = 2*kx+1;

	float* ker = (float*)malloc(sizeof(float)*num);
	if(!ker){
		printf("Allocation failed 5.....Exiting\n");
		exit(-1);
	}

	for(i = -kx; i <= kx; i++){
		ker[j] = exp(((float(-(i*i)))/(2*coeff*coeff)));
		sum = sum + ker[j];
		j++;
	}
	
	for( i = 0; i < num; i++){
		ker[i] = ker[i]/sum;
	}
	printf("\n");
	return ker;
}


Volume* 
warp_image(Volume* vol, float** vec)
{
	int i,x,y,z;
	int tempx1,tempy1,tempz1,jump;
	Volume* temp = (Volume*)malloc(sizeof(Volume));
	if(!temp){
		printf("Memory allocation failed for temporary volume...Exiting\n");
		exit(1);
	}
	
	for(i=0;i<3; i++){
		temp->dim[i] = vol->dim[i];
		temp->offset[i] = vol->offset[i];
		temp->pix_spacing[i] = vol->pix_spacing[i];
	}

	temp->npix = vol->npix;
	temp->pix_type = vol->pix_type;
	temp->xmax = vol->xmax;
	temp->xmin = vol->xmin;
	temp->ymax = vol->ymax;
	temp->ymin = vol->ymin;
	temp->zmax = vol->zmax;
	temp->zmin = vol->zmin;

	temp->img = (void*)malloc(sizeof(short)*temp->npix);
	memset (temp->img, -1000, sizeof(short)*temp->npix);

	short* temp2 = (short*)temp->img;
	short* temp1 = (short*)vol->img;

	i = 0;

	for(z=0; z < vol->dim[2]; z++)
		for(y=0; y < vol->dim[1]; y++)
			for(x=0; x < vol->dim[0]; x++){
				tempx1 = (x + (int)vec[0][i]);
				tempy1 = (y + (int)vec[1][i]);
				tempz1 = (z + (int)vec[2][i]);

				jump = i + (int)vec[0][i] + 
					((int)vec[1][i])*vol->dim[0] + 
					((int)vec[2][i])*vol->dim[0]*vol->dim[1];				
				
				if(!(tempx1 < 0 || tempx1 > vol->dim[0]-1 || tempy1 < 0 || tempy1 > vol->dim[1]-1 || 
					tempz1 < 0 || tempz1 > vol->dim[2]-1)){

					temp2[i] = temp1[jump];
				}
				else{
					temp2[i] = temp1[i];
				}
				i++;
			}
	return temp;
}


float**
computeNabla(Volume* vol1) 
{
#if defined (commentout)
    // assign dimensions
	float3 vol1_dimensions;
	vol1_dimensions.x = (float)vol1->dim[0];
	vol1_dimensions.y = (float)vol1->dim[1];
	vol1_dimensions.z = (float)vol1->dim[2];
    int size = vol1->npix;
    int tex_blocks_per_x = (int)ceil((double)vol1_dimensions.x/4);
    int tex_size = (int)ceil(sqrt((double)size/4.0));

    // create host copy of image
    short* temp=(short*)vol1->img;
    float* h_array=(float*)malloc(size*sizeof(float));
    for(int i=0;i<size;i++)
        h_array[i]=(float)temp[i];

    // create texture and copy host image onto texture    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray* cu_array;
    CUDA_SAFE_CALL( cudaMallocArray( &cu_array, &channelDesc, tex_size,tex_size)); 
    CUDA_SAFE_CALL( cudaMemcpyToArray( cu_array, 0, 0, h_array, size*sizeof(float), cudaMemcpyHostToDevice));
        
    // set texture parameters
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    CUDA_SAFE_CALL( cudaBindTexture( tex, cu_array, channelDesc));

    // setup execution parameters for an arbitrary sized volume
    dim3 threads(16,4,4);
    unsigned int blocks_in_y = ceil((double)(vol1_dimensions.x/4)/(double)threads.x);
    unsigned int y_in_z = ceil((double)vol1_dimensions.y/(double)threads.y);
    unsigned int rows_of_z = ceil((double)vol1_dimensions.z/(double)threads.z);
    dim3 grid(blocks_in_y * y_in_z,rows_of_z,1);
    
    // allocate device result memory
    float* d_odata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, size*sizeof(float)));
    
    // allocate host result memory
    float** h_result;
    h_result = (float**)malloc(3*sizeof(float*));
    h_result[0] = (float*)malloc(size*sizeof(float));
    h_result[1] = (float*)malloc(size*sizeof(float));
    h_result[2] = (float*)malloc(size*sizeof(float));

    // compute nabla_x
    compute_nabla_x<<<grid,threads>>>(d_odata, vol1_dimensions, size,tex_size,tex_blocks_per_x, vol1->pix_spacing[0]);
    CUT_CHECK_ERROR("Kernel execution failed");
    CUDA_SAFE_CALL( cudaMemcpy( h_result[0], d_odata, size*sizeof(float),cudaMemcpyDeviceToHost) );

    // compute nabla_y
    compute_nabla_y<<<grid,threads>>>(d_odata, vol1_dimensions, size,tex_size,tex_blocks_per_x, vol1->pix_spacing[1]);
    CUT_CHECK_ERROR("Kernel execution failed");
    CUDA_SAFE_CALL( cudaMemcpy( h_result[1], d_odata, size*sizeof(float),cudaMemcpyDeviceToHost) );

    // compute nabla_z
    compute_nabla_z<<<grid,threads>>>(d_odata, vol1_dimensions, size,tex_size,tex_blocks_per_x, vol1->pix_spacing[2]);
    CUT_CHECK_ERROR("Kernel execution failed");
    CUDA_SAFE_CALL( cudaMemcpy( h_result[2], d_odata, size*sizeof(float),cudaMemcpyDeviceToHost) );

    free(h_array);
    CUDA_SAFE_CALL(cudaFree(d_odata));
    CUDA_SAFE_CALL(cudaFreeArray(cu_array));
    return h_result;

#endif
    return 0;
}


float**
estimate_vector_field(Volume* vol1,Volume *vol2,float** der,int n_iter)
{
#if defined (commentout)
    // estimate vector field host code

    // declare important host variables
	int i,iter,p;
	float **vec;
    unsigned int size = vol1->npix;
    iter = n_iter;
    float* temp_holder;

	float3 dimen;
	dimen.x = (float)vol1->dim[0];
	dimen.y = (float)vol1->dim[1];
	dimen.z = (float)vol1->dim[2];

    float3 spacing;
    spacing.x = vol1->pix_spacing[0];
    spacing.y = vol1->pix_spacing[1];
    spacing.z = vol1->pix_spacing[2];

    // create timer
    unsigned int timer;
    cutCreateTimer(&timer);

	/* Allocate memory for the static and moving images */
	float* temp1 = (float*)malloc(sizeof(float)*size);
	if(!temp1){
		printf("Couldn't allocate memory for image 1...Exiting\n");
		exit(-1);
	}

	float* temp2 = (float*)malloc(sizeof(float)*size);
	if(!temp2){
		printf("Couldn't allocate memory for image 2...Exiting\n");
		exit(-1);
	}

	short* temp_vol1 = (short*)vol1->img;
	short* temp_vol2 = (short*)vol2->img;

	for(p=0; p < vol1->npix; p++){
		temp1[p] = (float)temp_vol1[p];
		temp2[p] = (float)temp_vol2[p];
	}

	/* Allocate memory for the vector or displacement fields */
	vec = (float**)malloc(3*sizeof(float*));
	if(!vec){
		printf("Memory allocation failed for stage 1 for current velocity..........Exiting\n");
		exit(1);
	}
	for(i=0; i < 3; i++){
		vec[i] = (float*)malloc(sizeof(float)*size);
		if(!vec[i]){
			printf("Memory allocation failed for stage 2, dimension %d for current velocity..........Exiting\n",i);
			exit(1);
		}
	}

    /* Allocate memory for statistics */
    // block_id = (float*)malloc(sizeof(float)*size);
	int* thread_id = (int*)malloc(sizeof(int)*size);
    int* voxel_id = (int*)malloc(sizeof(int)*size);

    // create a length 3 smoothing kernel
    float* ker;
    ker = create_ker( 1.0, 1);

    // ALLOCATE DEVICE MEMORY

    // NK: Allocate streams for statistics on the device 
    // float *block_id_on_device; 
    // CUDA_SAFE_CALL( cudaMalloc( (void**) &block_id_on_device, sizeof(float)*size));
    int *thread_id_on_device;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &thread_id_on_device, sizeof(int)*size));
    int *voxel_id_on_device;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &voxel_id_on_device, sizeof(int)*size));

    // allocate current vector on device
    float* vec_st_x;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &vec_st_x, sizeof(float)*size));
    float* vec_st_y;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &vec_st_y, sizeof(float)*size));
    float* vec_st_z;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &vec_st_z, sizeof(float)*size));

    // allocate previous vector on device
    float* pre_vec_st_x;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &pre_vec_st_x, sizeof(float)*size));
    float* pre_vec_st_y;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &pre_vec_st_y, sizeof(float)*size));
    float* pre_vec_st_z;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &pre_vec_st_z, sizeof(float)*size));

    // allocate nabla on device
    float* nabla_x;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &nabla_x, sizeof(float)*size));
    float* nabla_y;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &nabla_y, sizeof(float)*size));
    float* nabla_z;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &nabla_z, sizeof(float)*size));

    // allocate images on device
    float* static_image;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &static_image, sizeof(float)*size));
    float* moving_image;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &moving_image, sizeof(float)*size));

    // COPY HOST MEMORY TO DEVICE

    // create texture formats: channelDesc returns a float, channelDesc4 returns a float4
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    // copy nabla and bind to texture
    CUDA_SAFE_CALL( cudaMemcpy( nabla_x, der[0], size*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaBindTexture( tex_nabla_x, nabla_x, channelDesc, sizeof(float)*size,0));

    CUDA_SAFE_CALL( cudaMemcpy( nabla_y, der[1], size*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaBindTexture( tex_nabla_y, nabla_y, channelDesc,sizeof(float)*size,0));

    CUDA_SAFE_CALL( cudaMemcpy( nabla_z, der[2], size*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaBindTexture( tex_nabla_z, nabla_z, channelDesc,sizeof(float)*size,0));

    // copy images and bind to texture
    CUDA_SAFE_CALL( cudaMemcpy( static_image, temp1, size*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaBindTexture( tex_static_image, static_image, channelDesc,sizeof(float)*size,0));
    
    CUDA_SAFE_CALL( cudaMemcpy( moving_image, temp2, size*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaBindTexture( tex_moving_image, moving_image, channelDesc,sizeof(float)*size,0));


    // initialize device vectors pre_vec and vec and vectors for holding stats
    dim3 threads_i(16,16,1);
    dim3 grid_i(16,(int)ceil(((double)size)/((double)(256*16))),1);
    
    // k_initial_vector<<<grid_i,threads_i>>>(block_id, size);
    // CUT_CHECK_ERROR("Kernel execution failed");
    k_initial_vector_nk<<<grid_i,threads_i>>>(thread_id_on_device, size);
    CUT_CHECK_ERROR("Kernel execution failed");
    k_initial_vector_nk<<<grid_i,threads_i>>>(voxel_id_on_device, size);
    CUT_CHECK_ERROR("Kernel execution failed");

    k_initial_vector<<<grid_i,threads_i>>>(vec_st_x, size);
    CUT_CHECK_ERROR("Kernel execution failed");
    k_initial_vector<<<grid_i,threads_i>>>(vec_st_y, size);
    CUT_CHECK_ERROR("Kernel execution failed");
    k_initial_vector<<<grid_i,threads_i>>>(vec_st_z, size);
    CUT_CHECK_ERROR("Kernel execution failed");

    // setup execution parameters
    dim3 threads(2, 16, 2);
    unsigned int blocks_in_y = ceil((double)dimen.x/(double)threads.x);
    unsigned int y_in_z = ceil((double)dimen.x/(double)threads.y);
    unsigned int rows_of_z = ceil((double)dimen.z/(double)threads.z);
    dim3 grid(blocks_in_y * y_in_z,rows_of_z,1);

    // set up execution parameters for filter kernel
    int x_dimf = (int)dimen.x/4;
    dim3 threadsf(16,4,4);
    int x_in_yf = (int)ceil((double)x_dimf/threadsf.x);
    int y_in_zf = (int)ceil((double)dimen.y/threadsf.y);
    int z_totf = (int)ceil((double)dimen.z/threadsf.z);
    dim3 gridf(x_in_yf*y_in_zf,z_totf,1);

    for(int i=0;i<iter;i++)
    {
        printf(".");

        cutStartTimer(timer);
        // estimate vector field
        /* k_evf<<<grid,threads>>>(vec_st_x,
                                vec_st_y,
                                vec_st_z, 
					            dimen,
                                spacing,
					            size);
        CUT_CHECK_ERROR("Kernel execution failed");
        cutStopTimer(timer);
        */

        k_evf_nk<<<grid,threads>>>(vec_st_x,
                                vec_st_y,
                                vec_st_z,
                                thread_id_on_device,
                                voxel_id_on_device, 
					            dimen,
                                spacing,
					            size);
        CUT_CHECK_ERROR("Kernel execution failed");
        cutStopTimer(timer);

        //smooth vector field

        // smooth x direction of vec_st_x, vec_st_y and vec_st_z
        // bind data to be filtered to tex_filter, write out filtered data, unbind texture
        CUDA_SAFE_CALL( cudaBindTexture( tex_filter, vec_st_x, channelDesc4,sizeof(float)*size,0));
        gaussian_x<<<gridf,threadsf>>>(pre_vec_st_x, dimen, x_in_yf, y_in_zf, ker[0], ker[1], ker[0]+ker[1]);
        CUT_CHECK_ERROR("Kernel execution failed");
        CUDA_SAFE_CALL( cudaUnbindTexture(tex_filter));

        CUDA_SAFE_CALL( cudaBindTexture( tex_filter, vec_st_y, channelDesc4,sizeof(float)*size,0));
        gaussian_x<<<gridf,threadsf>>>(pre_vec_st_y, dimen, x_in_yf, y_in_zf, ker[0], ker[1], ker[0]+ker[1]);
        CUT_CHECK_ERROR("Kernel execution failed");
        CUDA_SAFE_CALL( cudaUnbindTexture(tex_filter));

        CUDA_SAFE_CALL( cudaBindTexture( tex_filter, vec_st_z, channelDesc4,sizeof(float)*size,0));
        gaussian_x<<<gridf,threadsf>>>(pre_vec_st_z, dimen, x_in_yf, y_in_zf, ker[0], ker[1], ker[0]+ker[1]);
        CUT_CHECK_ERROR("Kernel execution failed");
        CUDA_SAFE_CALL( cudaUnbindTexture(tex_filter));

        //smooth y direction of vector
        CUDA_SAFE_CALL( cudaBindTexture( tex_filter, pre_vec_st_x, channelDesc4,sizeof(float)*size,0));
        gaussian_y<<<gridf,threadsf>>>(vec_st_x, dimen, x_in_yf, y_in_zf, ker[0], ker[1], ker[0]+ker[1]);
        CUT_CHECK_ERROR("Kernel execution failed");
        CUDA_SAFE_CALL( cudaUnbindTexture(tex_filter));

        CUDA_SAFE_CALL( cudaBindTexture( tex_filter, pre_vec_st_y, channelDesc4,sizeof(float)*size,0));
        gaussian_y<<<gridf,threadsf>>>(vec_st_y, dimen, x_in_yf, y_in_zf, ker[0], ker[1], ker[0]+ker[1]);
        CUT_CHECK_ERROR("Kernel execution failed");
        CUDA_SAFE_CALL( cudaUnbindTexture(tex_filter));

        CUDA_SAFE_CALL( cudaBindTexture( tex_filter, pre_vec_st_z, channelDesc4,sizeof(float)*size,0));
        gaussian_y<<<gridf,threadsf>>>(vec_st_z, dimen, x_in_yf, y_in_zf, ker[0], ker[1], ker[0]+ker[1]);
        CUT_CHECK_ERROR("Kernel execution failed");
        CUDA_SAFE_CALL( cudaUnbindTexture(tex_filter));

        // smooth z direction of vector
        CUDA_SAFE_CALL( cudaBindTexture( tex_filter, vec_st_x, channelDesc4,sizeof(float)*size,0));
        gaussian_z<<<gridf,threadsf>>>(pre_vec_st_x, dimen, x_in_yf, y_in_zf, ker[0], ker[1], ker[0]+ker[1]);
        CUT_CHECK_ERROR("Kernel execution failed");
        CUDA_SAFE_CALL( cudaUnbindTexture(tex_filter));

        CUDA_SAFE_CALL( cudaBindTexture( tex_filter, vec_st_y, channelDesc4,sizeof(float)*size,0));
        gaussian_z<<<gridf,threadsf>>>(pre_vec_st_y, dimen, x_in_yf, y_in_zf, ker[0], ker[1], ker[0]+ker[1]);
        CUT_CHECK_ERROR("Kernel execution failed");
        CUDA_SAFE_CALL( cudaUnbindTexture(tex_filter));

        CUDA_SAFE_CALL( cudaBindTexture( tex_filter, vec_st_z, channelDesc4,sizeof(float)*size,0));
        gaussian_z<<<gridf,threadsf>>>(pre_vec_st_z, dimen, x_in_yf, y_in_zf, ker[0], ker[1], ker[0]+ker[1]);
        CUT_CHECK_ERROR("Kernel execution failed");
        CUDA_SAFE_CALL( cudaUnbindTexture(tex_filter));

        // swap pointers between vex_st and pre_vec_st for next iteration
        temp_holder=vec_st_x;
        vec_st_x=pre_vec_st_x;
        pre_vec_st_x=temp_holder;
        temp_holder=vec_st_y;
        vec_st_y=pre_vec_st_y;
        pre_vec_st_y=temp_holder;
        temp_holder=vec_st_z;
        vec_st_z=pre_vec_st_z;
        pre_vec_st_z=temp_holder;
    }
    printf("\nevf_texture time = %f \n",cutGetTimerValue(timer)/1000);
    cutDeleteTimer(timer);

    // convert vector from mm to voxels
    k_convert<<<grid,threads>>>(vec_st_x, vec_st_y,vec_st_z,pre_vec_st_x, pre_vec_st_y,pre_vec_st_z,
                                vol1->pix_spacing[0],vol1->pix_spacing[1],vol1->pix_spacing[2], dimen);
   
    // copy results from device to host
    CUDA_SAFE_CALL( cudaMemcpy( vec[0], pre_vec_st_x, sizeof(float)*size,cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy( vec[1], pre_vec_st_y, sizeof(float)*size,cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy( vec[2], pre_vec_st_z, sizeof(float)*size,cudaMemcpyDeviceToHost) );

    // copy stats from device to host
    CUDA_SAFE_CALL( cudaMemcpy( thread_id, thread_id_on_device, sizeof(int)*size,cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy( voxel_id, voxel_id_on_device, sizeof(int)*size,cudaMemcpyDeviceToHost) );

    for(i = 0; i < 64; i++)
       printf("Thread %d ---> Voxel %d \n", thread_id[i], voxel_id[i]); 
    getchar();
    
    // unbind textures
    CUDA_SAFE_CALL( cudaUnbindTexture(tex_nabla_x));
    CUDA_SAFE_CALL( cudaUnbindTexture(tex_nabla_y));
    CUDA_SAFE_CALL( cudaUnbindTexture(tex_nabla_z));
    CUDA_SAFE_CALL( cudaUnbindTexture(tex_moving_image));
    CUDA_SAFE_CALL( cudaUnbindTexture(tex_static_image));

    // cleanup memory
    free(temp1);
    free(temp2);
    free(thread_id);
    free(voxel_id);

    CUDA_SAFE_CALL(cudaFree(thread_id_on_device));
    CUDA_SAFE_CALL(cudaFree(voxel_id_on_device));

    CUDA_SAFE_CALL(cudaFree(vec_st_x));
    CUDA_SAFE_CALL(cudaFree(vec_st_y));
    CUDA_SAFE_CALL(cudaFree(vec_st_z));
    CUDA_SAFE_CALL(cudaFree(pre_vec_st_x));
    CUDA_SAFE_CALL(cudaFree(pre_vec_st_y));
    CUDA_SAFE_CALL(cudaFree(pre_vec_st_z));
    CUDA_SAFE_CALL(cudaFree(nabla_x));
    CUDA_SAFE_CALL(cudaFree(nabla_y));
    CUDA_SAFE_CALL(cudaFree(nabla_z));
    CUDA_SAFE_CALL(cudaFree(static_image));
    CUDA_SAFE_CALL(cudaFree(moving_image));

    return vec;
#endif
    return 0;
}



