/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */

// JAS 11.04.2010
// nvcc has the nasty limitation of not being able to
// use functions from other object files.  So in order
// to share __device__ functions amongst multiple .cu
// files, we resort to writing them in this somewhat
// kludgy "inc" file.

// Add the line:
//   #include "cuda_kernel_util.inc" 
//
// to the bottom of your CUDA .cu file to use these
// helper __device__ functions.

// ----------------------

// Note: n is the # of elements (NOT size_t)
template <typename T>
__device__ inline void
shared_memset (T* s, T c, int n)
{
    int tpb = (blockDim.x * blockDim.y * blockDim.z);
    int idx = (((blockDim.y * threadIdx.z) + threadIdx.y)
            * blockDim.x) + threadIdx.x;

    int b = (n + tpb - 1) / tpb;

    for (int i=0; i<b; i++) {
        if ( (idx + i*tpb) < n ) {
            s[idx + i*tpb] = c;
        }
    }
}

// Tell vim to use CUDA syntax highlighting
// vim:ft=cuda:
