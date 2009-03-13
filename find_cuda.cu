#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include "cutil.h"

int find_cuda (void)
{
	int devicecount;
	printf("Testing for CUDA...\n");
	cudaGetDeviceCount(&devicecount);

	if (devicecount == 0)
	{
		printf("No CUDA capable devices detected!\n");
		return 0;
	}

	printf ("CUDA detected!\n");
	// If desired, you could go on here to say more about the hardware.
	return 1;
}
