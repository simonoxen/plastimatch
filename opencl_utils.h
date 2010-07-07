/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _opencl_utils_h_
#define _opencl_utils_h_

#include "plm_config.h"
#if (OPENCL_FOUND)

#include <CL/cl.h>

typedef struct opencl_device Opencl_device;
struct opencl_device {
    cl_context context;
    cl_command_queue command_queue;
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
opencl_print_devices (void);
gpuit_EXPORT
void
opencl_open_device (Opencl_device *ocl_dev);
gpuit_EXPORT
void
opencl_close_device (Opencl_device *ocl_dev);

#if defined __cplusplus
}
#endif

#endif /* HAVE_OPENCL */
#endif
