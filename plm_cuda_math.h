/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cuda_math_h_
#define _cuda_math_h_

#include "plm_config.h"
#include <cuda.h>

/* Overloaded operators */
inline __host__ __device__ 
float3 
operator+ (float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ 
float3 
operator- (float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ 
float3 
operator* (float a, float3 b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ 
float3
operator* (float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ 
float3
operator/ (float a, float3 b)
{
    return make_float3(a / b.x, a / b.y, a / b.z);
}

inline __host__ __device__ 
int3
operator< (float3 a, float b)
{
    return make_int3 (a.x < b, a.y < b, a.z < b);
}

/* Misc functions */
inline __host__ __device__ 
float3
fabsf3 (float3 a)
{
    return make_float3 (fabsf(a.x), fabsf(a.y), fabsf(a.z));
}

inline __host__ __device__
void
swapf (float *a, float *b)
{
    float c = *a;
    *a = *b;
    *b = c;
}

inline __host__ __device__
void 
sortf (float *a, float *b)
{
    if (*a > *b) {
	swapf (a, b);
    }
}

inline __host__ __device__
void
sortf3 (float3 *a, float3 *b)
{
    sortf (&a->x, &b->x);
    sortf (&a->y, &b->y);
    sortf (&a->z, &b->z);
}

#endif
