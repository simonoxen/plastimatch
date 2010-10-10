/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include "bstring_util.h"
#include "file_util.h"
#include "opencl_util.h"
#include "plm_timer.h"
#include "print_and_exit.h"

//////////////////////////////////////////////////////////////////////////////
//! Custom Utility Functions
//////////////////////////////////////////////////////////////////////////////

cl_int
opencl_dump_platform_info (cl_platform_id platform)
{
    cl_int status;
    char buf[100];

    status = clGetPlatformInfo (
	platform, 
	CL_PLATFORM_NAME,
	sizeof (buf),
	buf,
	NULL);
    if (status != CL_SUCCESS) {
	return status;
    }
    printf ("  Name = %s\n", buf);

    status = clGetPlatformInfo (
	platform, 
	CL_PLATFORM_VENDOR,
	sizeof (buf),
	buf,
	NULL);
    if (status != CL_SUCCESS) {
	return status;
    }
    printf ("  Vendor = %s\n", buf);

    return CL_SUCCESS;
}

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
	printf ("Found %d platforms\n", num_platforms);
        status = clGetPlatformIDs (num_platforms, platform_list, NULL);
	if (status != CL_SUCCESS) {
	    print_and_exit ("Error in clGetPlatformIDs\n");
	}
	
        for (i = 0; i < num_platforms; i++)
	{
	    printf ("OpenCL platform [%d]\n", i);
	    status = opencl_dump_platform_info (platform_list[i]);
	    if (status != CL_SUCCESS) continue;

	    /* Choose first platform (?) */
	    if (!platform) {
		platform = platform_list[i];
	    }
	}
	free (platform_list);
    }
    return platform;
}

void
opencl_dump_devices (Opencl_device *ocl_dev)
{
    cl_int status;

    printf ("Num_devices = %d\n", ocl_dev->device_count);
    for (cl_uint i = 0; i < ocl_dev->device_count; i++) {
	char device_name[256];
	status = clGetDeviceInfo (
	    ocl_dev->devices[i], 
	    CL_DEVICE_NAME, 
	    sizeof(device_name), 
	    device_name, 
	    NULL);
	opencl_check_error (status, "Error with clGetDeviceInfo()");
	printf ("    Device [%d] = %s\n", i, device_name);
    }
}

/* Flavor a creates one context with multiple devices in it */
cl_int
opencl_create_context_a (Opencl_device *ocl_dev)
{
    cl_int status = 0;
    cl_context_properties cps[3];
    cl_context_properties* cprops;
    size_t device_list_size;

    if (ocl_dev->platform) {
	cps[0] = CL_CONTEXT_PLATFORM;
	cps[1] = (cl_context_properties) ocl_dev->platform;
	cps[2] = 0;
	cprops = cps;
    } else {
	cprops = NULL;
    }

    /* Create context from platform */
    ocl_dev->context_count = 1;
    ocl_dev->contexts = (cl_context*) malloc (sizeof(cl_context));
    ocl_dev->contexts[0] = clCreateContextFromType (
	cprops, 
	CL_DEVICE_TYPE_GPU, 
	NULL, 
	NULL, 
	&status);
    if (status != CL_SUCCESS) {
	print_and_exit ("Error in clCreateContextFromType\n");
    }

    /* Get size of device list */
    status = clGetContextInfo (
	ocl_dev->contexts[0], 
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

    /* Get the device list data */
    ocl_dev->device_count = device_list_size / sizeof(cl_device_id);
    ocl_dev->devices = (cl_device_id *) malloc (device_list_size);
    status = clGetContextInfo (
	ocl_dev->contexts[0], 
	CL_CONTEXT_DEVICES, 
	device_list_size, 
	ocl_dev->devices, 
	NULL);
    if (status != CL_SUCCESS) { 
	print_and_exit ("Error in clGetContextInfo\n");
    }

    /* Print out a little status about the devices */
    opencl_dump_devices (ocl_dev);

    /* Create command queue */
    ocl_dev->command_queues = (cl_command_queue*) malloc (
	sizeof(cl_command_queue));
    ocl_dev->command_queues[0] = clCreateCommandQueue (
	ocl_dev->contexts[0], 
	ocl_dev->devices[0], 
	0, 
	&status);
    if (status != CL_SUCCESS) { 
	print_and_exit ("Error in clCreateCommandQueue\n");
    }

    return CL_SUCCESS;
}

/* Flavor b creates multiple contexts, each with one device */
cl_int
opencl_create_context_b (Opencl_device *ocl_dev)
{
    cl_int status;
    cl_context_properties cps[3];
    cl_context_properties* cprops;

    if (ocl_dev->platform) {
	cps[0] = CL_CONTEXT_PLATFORM;
	cps[1] = (cl_context_properties) ocl_dev->platform;
	cps[2] = 0;
	cprops = cps;
    } else {
	cprops = NULL;
    }

    /* Get number of devices of type GPU on this platform */
    status = clGetDeviceIDs (
	ocl_dev->platform, 
	CL_DEVICE_TYPE_GPU, 
	0, 
	NULL, 
	&ocl_dev->device_count);
    if (status != CL_SUCCESS) return status;

    /* Get list of device ids */
    ocl_dev->devices = (cl_device_id *) malloc (
	ocl_dev->device_count * sizeof(cl_device_id));
    status = clGetDeviceIDs (
	ocl_dev->platform, 
	CL_DEVICE_TYPE_GPU, 
	ocl_dev->device_count, 
	ocl_dev->devices, 
	NULL);
    if (status != CL_SUCCESS) {
	return status;
    }

    /* Print out a little status about the devices */
    opencl_dump_devices (ocl_dev);

    /* Create context and command queue for each device */
    ocl_dev->context_count = ocl_dev->device_count;
    ocl_dev->contexts = (cl_context *) malloc (
	ocl_dev->context_count * sizeof(cl_context));
    ocl_dev->command_queues = (cl_command_queue *) malloc (
	ocl_dev->context_count * sizeof(cl_command_queue));
    for (cl_uint i = 0; i < ocl_dev->device_count; i++)
    {
	ocl_dev->contexts[i] = clCreateContext (
	    cprops, 
	    1, 
	    &ocl_dev->devices[i], 
	    NULL, 
	    NULL, 
	    &status);
	opencl_check_error (status, "clCreateContext");

	ocl_dev->command_queues[i] = clCreateCommandQueue (
	    ocl_dev->contexts[i], 
	    ocl_dev->devices[i], 
	    CL_QUEUE_PROFILING_ENABLE, 
	    &status);
	opencl_check_error (status, "clCreateContext");
    }

    return CL_SUCCESS;
}

void
opencl_open_device (Opencl_device *ocl_dev)
{
    cl_int status = 0;

    printf ("In opencl_open_device\n");

    /* Select platform */
    ocl_dev->platform = opencl_select_platform ();

    /* ATI examples suggest you can try to create a context and 
       command queue even if platform is NULL.  So we don't fail (yet) 
       if platform is NULL here.  */

    /* Create OpenCL context(s) and command queue(s) */
    //status = opencl_create_context_a (ocl_dev);
    status = opencl_create_context_b (ocl_dev);
}

void
opencl_close_device (Opencl_device *ocl_dev)
{
    cl_int status = 0;

    for (cl_uint i = 0; i < ocl_dev->context_count; i++) {
	status = clReleaseCommandQueue (ocl_dev->command_queues[i]);
	if (status != CL_SUCCESS) {
	    print_and_exit ("Error in clReleaseCommandQueue\n");
	}
	status = clReleaseContext (ocl_dev->contexts[i]);
	if (status != CL_SUCCESS) {
	    print_and_exit ("Error in clReleaseContext\n");
	}
    }
    free (ocl_dev->devices);
    free (ocl_dev->contexts);
    free (ocl_dev->command_queues);
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

cl_program
opencl_load_program (
    Opencl_device *ocl_dev, 
    const char* filename
)
{
    cl_int rc;
    cl_program program;
    CBString *buf;
    const char *buf_cstr;
    size_t len;

    /* Load the file contents into a string */
    buf = file_load (filename);

    /* Do the OpenCL stuff */
    buf_cstr = (const char*) (*buf);
    len = (size_t) buf->length ();
    program = clCreateProgramWithSource (
	ocl_dev->contexts[0], 
	1, 
	&buf_cstr, 
	&len, 
	&rc);
    opencl_check_error (rc, "Error calling clCreateProgramWithSource.");

    /* Free the string with file contents */
    delete buf;

    rc = clBuildProgram (program, 1, ocl_dev->devices, 
	NULL, NULL, NULL);
    if (rc != CL_SUCCESS) {
	opencl_dump_build_log (ocl_dev, program);
	opencl_check_error (rc, "Error calling clBuildProgram.");
    }

    return program;
}

cl_ulong 
opencl_timer (cl_event &event)
{
    cl_ulong start, end;

    clGetEventProfilingInfo (
	event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo (
	event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

    return (end - start);
}

void 
opencl_dump_build_log (Opencl_device *ocl_dev, cl_program program)
{
    cl_int rc;
    char buf[10240];

    rc = clGetProgramBuildInfo (program, ocl_dev->devices[0], 
	CL_PROGRAM_BUILD_LOG, sizeof(buf), buf, NULL);
    opencl_check_error (rc, "clGetProgramBuildInfo");
    printf ("Build log:\n%s\n", buf);
}

void
opencl_check_error (cl_int return_code, const char *msg)
{
    if (return_code != CL_SUCCESS) {
        print_and_exit ("OPENCL ERROR: %s (%d).\n", msg, return_code);
    }                         
}
