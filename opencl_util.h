/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _opencl_utils_h_
#define _opencl_utils_h_

#include "plm_config.h"
#if (OPENCL_FOUND)
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

//////////////////////////////////////////////////////////////////////////////
//! Custom Utility Functions
//////////////////////////////////////////////////////////////////////////////

typedef cl_mem Opencl_buf;

typedef struct opencl_device Opencl_device;
struct opencl_device {
    cl_platform_id platform;

    cl_uint context_count;
    cl_context *contexts;

    /* Each of these have device_count entries */
    cl_uint device_count;
    cl_device_id *devices;
    cl_command_queue *command_queues;
    cl_program *programs;
    cl_kernel *kernels;
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
cl_platform_id
opencl_select_platform (void);
gpuit_EXPORT
void
opencl_print_devices (void);
gpuit_EXPORT
cl_int
opencl_open_device (Opencl_device *ocl_dev);
gpuit_EXPORT
void
opencl_close_device (Opencl_device *ocl_dev);
gpuit_EXPORT
cl_ulong
opencl_timer (cl_event &event);
gpuit_EXPORT
void
opencl_check_error (cl_int return_code, const char *msg);
gpuit_EXPORT
void
opencl_load_programs (
    Opencl_device *ocl_dev, 
    const char* filename
);
gpuit_EXPORT
void 
opencl_dump_build_log (Opencl_device *ocl_dev, cl_program program);

gpuit_EXPORT
Opencl_buf* 
opencl_buf_create (
    Opencl_device *ocl_dev, 
    cl_mem_flags flags, 
    size_t buffer_size, 
    void *buffer
);
gpuit_EXPORT
void
opencl_buf_read (
    Opencl_device *ocl_dev, 
    Opencl_buf* ocl_buf, 
    size_t buffer_size, 
    void *buffer
);
gpuit_EXPORT
void
opencl_buf_write (
    Opencl_device *ocl_dev, 
    Opencl_buf* ocl_buf, 
    size_t buffer_size, 
    void *buffer
);
gpuit_EXPORT
void
opencl_kernel_create (
    Opencl_device *ocl_dev, 
    const char *kernel_name
);
gpuit_EXPORT
void
opencl_set_kernel_args (
    Opencl_device *ocl_dev, 
    ...
);
gpuit_EXPORT
void
opencl_kernel_enqueue (
    Opencl_device *ocl_dev, 
    size_t global_work_size, 
    size_t local_work_size
);

#if defined __cplusplus
}
#endif

#endif /* HAVE_OPENCL */
#endif
