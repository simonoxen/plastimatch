/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "clamp.h"
#include "logfile.h"
#include "plm_int.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "volume.h"
#include "volume_header.h"
#include "volume_conv.h"

static void
pixel_conv (
    float* img_out,
    const float *img_in,
    const float *img_ker,
    const plm_long *dim_in,
    const plm_long *dim_ker,
    const plm_long *ker_hw,
    const plm_long *ijk_out)
{
    /* GCS FIX: This could be made faster by adding preamble and 
       postable loops rather than clamping */
    plm_long ijk_ker[3];  /* kernel ijk of kernel */
    plm_long ijk_in[3];      /* image ijk of kernel overlaid */
    plm_long out_v = volume_index (dim_in, ijk_out);

    for (ijk_ker[2] = 0; ijk_ker[2] < dim_ker[2]; ijk_ker[2]++) {
        ijk_in[2] = ijk_out[2] + ijk_ker[2] - ker_hw[2];
        CLAMP (ijk_in[2], 0, dim_in[2]-1);
        for (ijk_ker[1] = 0; ijk_ker[1] < dim_ker[1]; ijk_ker[1]++) {
            ijk_in[1] = ijk_out[1] + ijk_ker[1] - ker_hw[1];
            CLAMP (ijk_in[1], 0, dim_in[1]-1);
            for (ijk_ker[0] = 0; ijk_ker[0] < dim_ker[0]; ijk_ker[0]++) {
                ijk_in[0] = ijk_out[0] + ijk_ker[0] - ker_hw[0];
                CLAMP (ijk_in[0], 0, dim_in[0]-1);
                plm_long ker_v = volume_index (dim_ker, ijk_ker);
                plm_long in_v = volume_index (dim_in, ijk_in);
                img_out[out_v] += img_ker[ker_v] * img_in[in_v];
            }
        }
    }
}

Volume::Pointer
volume_conv (
    const Volume::Pointer& vol_in,
    const Volume::Pointer& ker_in)
{
    Volume::Pointer vol_out = vol_in->clone_empty();
    const float *img_in = vol_in->get_raw<float> ();
    const float *img_ker = ker_in->get_raw<float> ();
    float *img_out = vol_out->get_raw<float> ();
    const plm_long* dim_in = vol_in->dim;
    const plm_long* dim_ker = ker_in->dim;

    /* Compute kernel half-width */
    plm_long ker_hw[3];
    for (int d = 0; d < 3; d++) {
        ker_hw[d] = dim_ker[d] / 2;
    }

#pragma omp parallel for 
    LOOP_Z_OMP (k, vol_in) {
        plm_long ijk_out[3];
        ijk_out[2] = k;
        for (ijk_out[1] = 0; ijk_out[1] < vol_in->dim[1]; ijk_out[1]++) {
            for (ijk_out[0] = 0; ijk_out[0] < vol_in->dim[0]; ijk_out[0]++) {
                pixel_conv (img_out, img_in, img_ker, dim_in, dim_ker, 
                    ker_hw, ijk_out);
            }
        }
    }

    return vol_out;
}
