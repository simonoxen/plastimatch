#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include <cutil.h>

extern "C" int
cuda_probe (void)
{
    int devicecount;
    printf ("Testing for CUDA...\n");
    cudaGetDeviceCount (&devicecount);

    if (devicecount == 0)
    {
	printf("Suitable CUDA environment not detected!\n");
	return 0;
    }

    // It is possible at this point that devicecount = 1 and still be
    // without an actual CUDA device.  CUDA 2.0 and 2.1 exhibit this
    // behavior.  Apparently 2.x will detect an emulator device and
    // throw a 1 by reference when you call cudaGetDeviceCount().
    // You are apparently able to distinguish between an actual
    // CUDA device and the emulator by checking the major and minor
    // revision numbers on the compute capability.  Emulated devices
    // are supposed to return 9999 for both major and minor revision
    // numbers.  Some, however, report that while this is the behavior
    // for CUDA 2.0, CUDA 2.1 returns different nonsensical numbers
    // when the detected device is emulated.  Therefore, the best
    // solution (until the behavior is standardised across releases)
    // is to specifically check for compute capabilities we KNOW are
    // working with Plastimatch.
    //

    // Get CUDA device properties.
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    if(props.major == 1)
    {
	if(props.minor == 0)
	    printf ("Detected CUDA!  Compute Capability 1.0");
	else if(props.minor == 1)
	    printf ("Detected CUDA!  Compute Capability 1.1");
	else if(props.minor == 2)
	    printf ("Detected CUDA!  Compute Capability 1.2");
	else
	{
	    printf ("Suitable CUDA environment not detected!\n");
	    return 0;
	}
	return 1;
    }

    // Unless proven otherwise, we assume no CUDA.
    return 0;
}
