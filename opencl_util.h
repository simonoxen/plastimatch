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

typedef struct opencl_device Opencl_device;
struct opencl_device {
    cl_context context;
    size_t device_list_size;
    cl_device_id *devices;
    cl_command_queue command_queue;
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
void
opencl_open_device (Opencl_device *ocl_dev);
gpuit_EXPORT
void
opencl_close_device (Opencl_device *ocl_dev);
gpuit_EXPORT
cl_ulong
executionTime(cl_event &event);
gpuit_EXPORT
void
opencl_check_error (cl_int return_code, const char *msg);
gpuit_EXPORT
cl_program
opencl_load_program (
    Opencl_device *ocl_dev, 
    const char* filename
);

#if defined __cplusplus
}
#endif

#endif /* HAVE_OPENCL */
#endif
