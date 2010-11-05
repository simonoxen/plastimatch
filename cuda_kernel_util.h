/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cuda_kernel_util_h_
#define _cuda_kernel_util_h_
//
// NOTE: Cannot be included in C or C++ files due to
//       special CUDA types such as int4, dim3, etc.
//       Can only be included in CUDA files.
//
// NOTE: __device__ function definitions live in
//       cuda_kernel_util.inc due to nvcc limitations.

#include "plm_config.h"
#include <cuda.h>

#define GRID_LIMIT_X 65535
#define GRID_LIMIT_Y 65535

template <typename T>
__device__ inline void
shared_memset (T* s, T c, int n);

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
int
CUDA_exec_conf_1tpe (
    dim3 *dimGrid,          // OUTPUT: Grid  dimensions
    dim3 *dimBlock,         // OUTPUT: Block dimensions
    int num_threads,        // INPUT: Total # of threads
    int threads_per_block,  // INPUT: Threads per block
    bool negotiate          // INPUT: Is threads per block negotiable?
);

gpuit_EXPORT
void
CUDA_exec_conf_1bpe (
    dim3 *dimGrid,           // OUTPUT: Grid  dimensions
    dim3 *dimBlock,          // OUTPUT: Block dimensions
    int num_blocks,          //  INPUT: Number of blocks
    int threads_per_block    //  INPUT: Threads per block
);

#if defined __cplusplus
}
#endif

#endif
