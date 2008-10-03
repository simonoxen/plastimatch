/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <cuda.h>

#define SIZE 256

int
main (int argc, char* argv)
{
    int A[SIZE], B[SIZE], C[SIZE];
    int *Ad, *Bd, *Cd;

    cudaMalloc ((void**) &Ad, sizeof(float)*SIZE);

    cudaFree (Ad);
}
