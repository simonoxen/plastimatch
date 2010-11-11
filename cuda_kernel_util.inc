/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */

// JAS 2010.11.04
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
shared_memset (
    T* s,
    T c,
    int n
)
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

// JAS 2010.11.11
// You can use this hack to get floating point atomicAdd for devices with
// compute less than 2.0.  However, note that atomicExch() can be performed on
// both global and shared memory on a 1.2 device, but a 1.1 device can only do
// atomicExch() on global memory.  So, be careful.  If you use this function to
// operate on shared memory, surround the call with an #if __CUDA_ARCH__ >= 120
__device__ inline void
atomic_add_float (
    float* addr,
    float val
)
{
#if __CUDA_ARCH__ >= 200
    atomicAdd (addr, val);
#elif __CUDA_ARCH__ >= 110 && __CUDA_ARCH__ < 200
    float old;
    bool success = false;
    while (!success) {
        old = atomicExch(addr, -1.0f);
        if (old != -1.0f) {
            success = true;
            old += val;
            atomicExch(addr, old);
        }
            __threadfence();
    }
#else
    /* JAS 2010.11.11
       I don't think there is a way to do this
       without memory atomics... so make sure that this case never happens.  We
       just have it so that things will compile for the "old skool" CUDA
       devices.
    */
    ;
#endif
}

// Tell vim to use CUDA syntax highlighting
// vim:ft=cuda:
