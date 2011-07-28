/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "threading.h"

Threading
threading_parse (const char *string)
{
    if (!strcmp (string, "CPU") || !strcmp (string, "cpu")) {
	return THREADING_CPU_OPENMP;
    }
    if (!strcmp (string, "CUDA") || !strcmp (string, "cuda")) {
	return THREADING_CUDA;
    }
    if (!strcmp (string, "GPU") || !strcmp (string, "gpu")) {
	return THREADING_OPENCL;
    }
    if (!strcmp (string, "OPENCL") || !strcmp (string, "opencl")) {
	return THREADING_OPENCL;
    }
    if (!strcmp (string, "OPENMP") || !strcmp (string, "openmp")) {
	return THREADING_CPU_OPENMP;
    }
    if (!strcmp (string, "SINGLE") || !strcmp (string, "single")) {
	return THREADING_CPU_SINGLE;
    }

    /* Default case */
    return THREADING_UNKNOWN;
}
