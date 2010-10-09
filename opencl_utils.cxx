/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "opencl_utils.h"
#include "plm_timer.h"
#include "print_and_exit.h"

//////////////////////////////////////////////////////////////////////////////
//! Custom Utility Functions
//////////////////////////////////////////////////////////////////////////////

cl_platform_id
opencl_select_platform (void)
{
    cl_int status = 0;
    cl_uint num_platforms;
    cl_platform_id platform = NULL;

    status = clGetPlatformIDs (0, NULL, &num_platforms);
    if (status != CL_SUCCESS) {
	print_and_exit ("Error in clGetPlatformIDs\n");
    }
    if (num_platforms > 0) {
        unsigned int i;
        cl_platform_id* platform_list = (cl_platform_id*) malloc (
	    sizeof (cl_platform_id) * num_platforms);
        status = clGetPlatformIDs (num_platforms, platform_list, NULL);
	if (status != CL_SUCCESS) {
	    print_and_exit ("Error in clGetPlatformIDs\n");
	}
	
        for (i = 0; i < num_platforms; i++) {
	    char pbuff[100];
            status = clGetPlatformInfo (
		platform_list[i],
		CL_PLATFORM_VENDOR,
		sizeof (pbuff),
		pbuff,
		NULL);
	    platform = platform_list[i];
	    printf ("OpenCL platform [%d] = %s\n", i, pbuff);
	}
	free (platform_list);
    }
    return platform;
}

void
opencl_open_device (Opencl_device *ocl_dev)
{
    cl_int status = 0;
    size_t device_list_size;
    cl_device_id *devices;
    cl_platform_id platform;
    cl_context_properties cps[3];
    cl_context_properties* cprops;
    Timer timer;

    /* Select platform */
    platform = opencl_select_platform ();
    if (platform) {
	cps[0] = CL_CONTEXT_PLATFORM;
	cps[1] = (cl_context_properties) platform;
	cps[2] = 0;
	cprops = cps;
    } else {
	cprops = NULL;
    }

    /* Create context */
    plm_timer_start (&timer);
    ocl_dev->context = clCreateContextFromType (
	cprops, 
	CL_DEVICE_TYPE_GPU, 
	NULL, 
	NULL, 
	&status);
    if (status != CL_SUCCESS) {  
	print_and_exit ("Error in clCreateContextFromType\n");
    }
    printf ("Create context: %f sec\n", plm_timer_report (&timer));

    /* Get size of device list */
    plm_timer_start (&timer);
    status = clGetContextInfo (
	ocl_dev->context, 
	CL_CONTEXT_DEVICES, 
	0, 
	NULL, 
	&device_list_size);
    if (status != CL_SUCCESS) {
	print_and_exit ("Error in clGetContextInfo\n");
    }
    if (device_list_size == 0) {
	print_and_exit ("No devices found (clGetContextInfo)\n");
    }

    devices = (cl_device_id *) malloc (device_list_size);

    /* Get the device list data */
    status = clGetContextInfo (
	ocl_dev->context, 
	CL_CONTEXT_DEVICES, 
	device_list_size, 
	devices, 
	NULL);
    if (status != CL_SUCCESS) { 
	print_and_exit ("Error in clGetContextInfo\n");
    }
    printf ("Get context info: %f sec\n", plm_timer_report (&timer));

    /* Create OpenCL command queue */
    plm_timer_start (&timer);
    ocl_dev->command_queue = clCreateCommandQueue (
	ocl_dev->context, 
	devices[0], 
	0, 
	&status);
    if (status != CL_SUCCESS) { 
	print_and_exit ("Error in clCreateCommandQueue\n");
    }
    printf ("Create command queue: %f sec\n", plm_timer_report (&timer));
}

void
opencl_close_device (Opencl_device *ocl_dev)
{
    cl_int status = 0;

    status = clReleaseCommandQueue (ocl_dev->command_queue);
    if (status != CL_SUCCESS) {
	print_and_exit ("Error in clReleaseCommandQueue\n");
    }
    status = clReleaseContext (ocl_dev->context);
    if (status != CL_SUCCESS) {
	print_and_exit ("Error in clReleaseContext\n");
    }
}

#if defined (commentout)
    /////////////////////////////////////////////////////////////////
    // Create OpenCL memory buffers
    /////////////////////////////////////////////////////////////////
    inputBuffer = clCreateBuffer (
	context, 
	CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
	sizeof(cl_uint) * width,
	input, 
	&status);
    if(status != CL_SUCCESS) 
    { 
	std::cout<<"Error: clCreateBuffer (inputBuffer)\n";
	return 1;
    }

    outputBuffer = clCreateBuffer(
	context, 
	CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
	sizeof(cl_uint) * width,
	output, 
	&status);
    if(status != CL_SUCCESS) 
    { 
	std::cout<<"Error: clCreateBuffer (outputBuffer)\n";
	return 1;
    }
#endif


#if defined (commentout)
    /////////////////////////////////////////////////////////////////
    // Load CL file, build CL program object, create CL kernel object
    /////////////////////////////////////////////////////////////////
    const char * filename  = "opencl_test.cl";
    std::string  sourceStr = convertToString(filename);
    const char * source    = sourceStr.c_str();
    size_t sourceSize[]    = { strlen(source) };

    program = clCreateProgramWithSource(
	context, 
	1, 
	&source,
	sourceSize,
	&status);
    if(status != CL_SUCCESS) 
    { 
	std::cout<<
	    "Error: Loading Binary into cl_program \
			   (clCreateProgramWithBinary)\n";
	return 1;
    }

    /* create a cl program executable for all the devices specified */
    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    if(status != CL_SUCCESS) 
    { 
	std::cout<<"Error: Building Program (clBuildProgram)\n";
	return 1; 
    }

    /* get a kernel object handle for a kernel with the given name */
    kernel = clCreateKernel(program, "templateKernel", &status);
    if(status != CL_SUCCESS) 
    {  
	std::cout<<"Error: Creating Kernel from program. (clCreateKernel)\n";
	return 1;
    }

    return 0;
}

#endif

char* 
opencl_load_program (
    Opencl_device *ocl_dev, 
    const char* filename
)
{
    char *buf;

#if defined (commentout)
    /* Load the file contents into a string */
    buf = file_load_with_len (filename, &len);

    /* Do the OpenCL stuff */
    program = clCreateProgramWithSource (
	ocl_dev->context, 
	1, 
	&source,
	sourceSize,
	&status);
#endif
}

cl_ulong executionTime(cl_event &event)
{
	cl_ulong start, end;

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

	return (end - start);
}

void
opencl_check_error (cl_int return_code, const char *msg)
{
    if (return_code != CL_SUCCESS) {
        print_and_exit ("OPENCL ERROR: %s (%d).\n", msg, return_code);
    }                         
}
