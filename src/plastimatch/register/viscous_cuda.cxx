/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include "viscous_cuda.h"

int 
CUDA_viscous_main (
    int argc, 
    char** argv
)
{
    return CUDA_viscous (argc, argv);
}
