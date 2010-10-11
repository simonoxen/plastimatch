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

void
opencl_device_info (
    cl_device_id device, 
    cl_device_info param_name, 
    size_t param_value_size,  
    void *param_value,  
    size_t *param_value_size_ret
)
{
    cl_int status;
    status = clGetDeviceInfo (
	device, 
	param_name, 
	param_value_size,  
	param_value,  
	param_value_size_ret);
    opencl_check_error (status, "clGetDeviceInfo");
}

void
opencl_dump_device_info (cl_device_id device)
{
    char param_string[1024];
    cl_bool param_bool;

    opencl_device_info (
	device, 
	CL_DEVICE_VENDOR, 
	sizeof(param_string), 
	param_string, 
	NULL);
    printf ("  CL_DEVICE_VENDOR = %s\n", param_string);

    opencl_device_info (
	device, 
	CL_DEVICE_NAME, 
	sizeof(param_string), 
	param_string, 
	NULL);
    printf ("  CL_DEVICE_NAME = %s\n", param_string);

    opencl_device_info (
	device, 
	CL_DEVICE_AVAILABLE, 
	sizeof (cl_bool), 
	&param_bool, 
	NULL);
    printf ("  CL_DEVICE_AVAILABLE = %d\n", param_bool);

    opencl_device_info (
	device, 
	CL_DEVICE_VERSION, 
	sizeof(param_string), 
	param_string, 
	NULL);
    printf ("  CL_DEVICE_VERSION = %s\n", param_string);

    opencl_device_info (
	device, 
	CL_DRIVER_VERSION, 
	sizeof(param_string), 
	param_string, 
	NULL);
    printf ("  CL_DRIVER_VERSION = %s\n", param_string);

    opencl_device_info (
	device, 
	CL_DEVICE_IMAGE_SUPPORT, 
	sizeof (cl_bool), 
	&param_bool, 
	NULL);
    printf ("  CL_DEVICE_IMAGE_SUPPORT = %d\n", param_bool);
}

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
    printf ("Num_devices = %d\n", ocl_dev->device_count);
    for (cl_uint i = 0; i < ocl_dev->device_count; i++) {
	printf ("OpenCL device [%d]\n", i);
	opencl_dump_device_info (ocl_dev->devices[i]);
    }
}

/* Create one command queue for each device */
cl_int
opencl_create_command_queues (Opencl_device *ocl_dev)
{
    cl_int status;

    ocl_dev->command_queues = (cl_command_queue *) malloc (
	ocl_dev->device_count * sizeof(cl_command_queue));
    for (cl_uint i = 0; i < ocl_dev->device_count; i++)
    {
	cl_uint cxt_no;

	/* Find the right context depending if method a or b was used */
	if (ocl_dev->context_count == 1) {
	    cxt_no = 0;
	} else {
	    cxt_no = i;
	}

	ocl_dev->command_queues[i] = clCreateCommandQueue (
	    ocl_dev->contexts[cxt_no], 
	    ocl_dev->devices[i], 
	    CL_QUEUE_PROFILING_ENABLE, 
	    &status);
	if (status != CL_SUCCESS) {
	    return status;
	}
    }
    return CL_SUCCESS;
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
    }

    return CL_SUCCESS;
}

cl_int
opencl_open_device (Opencl_device *ocl_dev)
{
    cl_int status = 0;

    memset (ocl_dev, 0, sizeof(Opencl_device));

    /* Select platform */
    ocl_dev->platform = opencl_select_platform ();

    /* ATI examples suggest you can try to create a context and 
       command queue even if platform is NULL.  So we don't fail (yet) 
       if platform is NULL here.  */

    /* Create contexts (there are two versions of this function: a, b) */
    //status = opencl_create_context_a (ocl_dev);
    status = opencl_create_context_b (ocl_dev);
    if (status != CL_SUCCESS) {
	return status;
    }

    /* Create command queues */
    status = opencl_create_command_queues (ocl_dev);
    return status;
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

void
opencl_load_programs (
    Opencl_device *ocl_dev, 
    const char* filename
)
{
    cl_int status;
    CBString *buf;
    const char *buf_cstr;
    size_t len;

    /* Load the file contents into a string */
    buf = file_load (filename);

    /* Load and compile the programs */
    buf_cstr = (const char*) (*buf);
    len = (size_t) buf->length ();
    ocl_dev->programs = (cl_program*) malloc (
	ocl_dev->context_count * sizeof(cl_program));
    for (cl_uint i = 0; i < ocl_dev->context_count; i++) {
	ocl_dev->programs[i] = clCreateProgramWithSource (
	    ocl_dev->contexts[i], 
	    1, 
	    &buf_cstr, 
	    &len, 
	    &status);
	opencl_check_error (status, 
	    "Error calling clCreateProgramWithSource.");

	/* Here we need to find the devices associated with this context,
	   which depends on if method a or b was used. */
	if (ocl_dev->context_count == 1) {
	    status = clBuildProgram (
		ocl_dev->programs[i], 
		ocl_dev->device_count, 
		ocl_dev->devices, 
		NULL, 
		NULL, 
		NULL);
	} else {
	    status = clBuildProgram (
		ocl_dev->programs[i], 
		1, 
		&ocl_dev->devices[i], 
		NULL, 
		NULL, 
		NULL);
	}
	if (status != CL_SUCCESS) {
	    opencl_dump_build_log (ocl_dev, ocl_dev->programs[i]);
	    opencl_check_error (status, "Error calling clBuildProgram.");
	}
    }

    /* Free the string with file contents */
    delete buf;
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

const char*
opencl_error_string (cl_int status)
{
    static const char* error_strings[] = {
        "CL_SUCCESS",
        "CL_DEVICE_NOT_FOUND",
        "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES",
        "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE",
        "CL_MAP_FAILURE",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "CL_INVALID_VALUE",
        "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE",
        "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM",
        "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION",
        "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS",
        "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION",
        "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
    };
    status = -status;
    if (status < 0 || status >= (cl_int) sizeof(error_strings)) {
	return "";
    }
    return error_strings[status];
}

void
opencl_check_error (cl_int status, const char *msg)
{
    if (status != CL_SUCCESS) {
        print_and_exit ("OPENCL ERROR: %s (%d,%s).\n", 
	    msg, status, opencl_error_string (status));
    }                         
}
