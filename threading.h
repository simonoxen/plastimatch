/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _threading_h_
#define _threading_h_

enum Threading {
    THREADING_CPU_SINGLE,
    THREADING_CPU_OPENMP,
    THREADING_BROOK,
    THREADING_CUDA,
    THREADING_OPENCL
};

#endif
