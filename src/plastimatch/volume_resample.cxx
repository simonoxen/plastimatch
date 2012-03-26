/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interpolate.h"
#include "math_util.h"
#include "plm_int.h"
#include "print_and_exit.h"
#include "volume.h"
#include "volume_resample.h"

/* Nearest neighbor interpolation */
static Volume*
volume_resample_float_nn (
    Volume* vol_in, plm_long* dim, float* offset, float* spacing)
{
    plm_long i, j, k, v;
    float x, y, z;
    float x_in, y_in, z_in;
    plm_long xidx, yidx, zidx;
    Volume* vol_out;
    float *in_img, *out_img;
    float val;
    float default_val = 0.0f;

    vol_out = new Volume (dim, offset, spacing, vol_in->direction_cosines, 
        PT_FLOAT, 1);
    in_img = (float*) vol_in->img;
    out_img = (float*) vol_out->img;

    for (k = 0, v = 0, z = offset[2]; k < dim[2]; k++, z += spacing[2]) {
        z_in = (z - vol_in->offset[2]) / vol_in->spacing[2];
        zidx = ROUND_INT (z_in);
        for (j = 0, y = offset[1]; j < dim[1]; j++, y += spacing[1]) {
            y_in = (y - vol_in->offset[1]) / vol_in->spacing[1];
            yidx = ROUND_INT (y_in);
            for (i = 0, x = offset[0]; i < dim[0]; i++, x += spacing[0], v++) {
                x_in = (x - vol_in->offset[0]) / vol_in->spacing[0];
                xidx = ROUND_INT (x_in);
                if (zidx < 0 || zidx >= vol_in->dim[2] || yidx < 0 || yidx >= vol_in->dim[1] || xidx < 0 || xidx >= vol_in->dim[0]) {
                    val = default_val;
                } else {
                    int idx = zidx*vol_in->dim[1]*vol_in->dim[0] + yidx*vol_in->dim[0] + xidx;
                    val = in_img[idx];
                }
                out_img[v] = val;
            }
        }
    }

    return vol_out;
}

/* Linear interpolation */
static Volume*
volume_resample_float_li (
    Volume* vol_in, plm_long* dim, float* offset, float* spacing)
{
    plm_long i, j, k, v;
    float x, y, z;
    //float x_in, y_in, z_in;
    plm_long xidx, yidx, zidx;
    Volume* vol_out;
    float *in_img, *out_img;
    float val;
    float default_val = 0.0f;
    float ijk[3];

    vol_out = new Volume (dim, offset, spacing, vol_in->direction_cosines, 
        PT_FLOAT, 1);
    in_img = (float*) vol_in->img;
    out_img = (float*) vol_out->img;

    for (k = 0, v = 0, z = offset[2]; k < dim[2]; k++, z += spacing[2]) {
        ijk[2] = (z - vol_in->offset[2]) / vol_in->spacing[2];
        zidx = ROUND_INT (ijk[2]);
        for (j = 0, y = offset[1]; j < dim[1]; j++, y += spacing[1]) {
            ijk[1] = (y - vol_in->offset[1]) / vol_in->spacing[1];
            yidx = ROUND_INT (ijk[1]);
            for (i = 0, x = offset[0]; i < dim[0]; i++, x += spacing[0], v++) {
                ijk[0] = (x - vol_in->offset[0]) / vol_in->spacing[0];
                xidx = ROUND_INT (ijk[0]);
                if (zidx < 0 || zidx >= vol_in->dim[2] || 
                    yidx < 0 || yidx >= vol_in->dim[1] || 
                    xidx < 0 || xidx >= vol_in->dim[0])
                {
                    val = default_val;
                } else {
                    plm_long ijk_floor[3];
                    plm_long ijk_round[3];
                    float li_1[3], li_2[3];
                    plm_long idx_floor;

                    // Compute linear interpolation fractions
                    li_clamp_3d (ijk, ijk_floor, ijk_round,
                        li_1, li_2, vol_in);

                    // Find linear indices for moving image
                    idx_floor = volume_index (vol_in->dim, ijk_floor);

                    // Calc. moving voxel intensity via linear interpolation
                    LI_VALUE (
                        val, 
                        li_1[0], li_2[0],
                        li_1[1], li_2[1],
                        li_1[2], li_2[2],
                        idx_floor,
                        in_img, vol_in
                    );
                }
                out_img[v] = val;
            }
        }
    }

    return vol_out;
}

/* Nearest neighbor interpolation */
static Volume*
volume_resample_vf_float_interleaved (
    Volume* vol_in, plm_long* dim, 
    float* offset, float* spacing)
{
    plm_long d, i, j, k, v;
    float x, y, z;
    float x_in, y_in, z_in;
    plm_long xidx, yidx, zidx;
    Volume* vol_out;
    float *in_img, *out_img;
    float* val;
    float default_val[3] = { 0.0f, 0.0f, 0.0f };

    vol_out = new Volume (dim, offset, spacing, vol_in->direction_cosines, 
        PT_VF_FLOAT_INTERLEAVED, 3);
    in_img = (float*) vol_in->img;
    out_img = (float*) vol_out->img;

    for (k = 0, v = 0, z = offset[2]; k < dim[2]; k++, z += spacing[2]) {
        z_in = (z - vol_in->offset[2]) / vol_in->spacing[2];
        zidx = ROUND_INT (z_in);
        for (j = 0, y = offset[1]; j < dim[1]; j++, y += spacing[1]) {
            y_in = (y - vol_in->offset[1]) / vol_in->spacing[1];
            yidx = ROUND_INT (y_in);
            for (i = 0, x = offset[0]; i < dim[0]; i++, x += spacing[0]) {
                x_in = (x - vol_in->offset[0]) / vol_in->spacing[0];
                xidx = ROUND_INT (x_in);
                if (zidx < 0 || zidx >= vol_in->dim[2] || yidx < 0 || yidx >= vol_in->dim[1] || xidx < 0 || xidx >= vol_in->dim[0]) {
                    val = default_val;
                } else {
                    plm_long idx = zidx*vol_in->dim[1]*vol_in->dim[0] + yidx*vol_in->dim[0] + xidx;
                    val = &in_img[idx*3];
                }
                for (d = 0; d < 3; d++, v++) {
                    out_img[v] = val[d];
                }
            }
        }
    }

    return vol_out;
}

/* Nearest neighbor interpolation */
static Volume*
volume_resample_vf_float_planar (
    Volume* vol_in, plm_long* dim, 
    float* offset, float* spacing)
{
    plm_long d, i, j, k, v;
    float x, y, z;
    float x_in, y_in, z_in;
    plm_long xidx, yidx, zidx;
    Volume* vol_out;
    float **in_img, **out_img;

    vol_out = new Volume (dim, offset, spacing, vol_in->direction_cosines, 
        PT_VF_FLOAT_PLANAR, 3);
    in_img = (float**) vol_in->img;
    out_img = (float**) vol_out->img;

    for (k = 0, v = 0, z = offset[2]; k < dim[2]; k++, z += spacing[2]) {
        z_in = (z - vol_in->offset[2]) / vol_in->spacing[2];
        zidx = ROUND_INT (z_in);
        for (j = 0, y = offset[1]; j < dim[1]; j++, y += spacing[1]) {
            y_in = (y - vol_in->offset[1]) / vol_in->spacing[1];
            yidx = ROUND_INT (y_in);
            for (i = 0, x = offset[0]; i < dim[0]; i++, x += spacing[0], v++) {
                x_in = (x - vol_in->offset[0]) / vol_in->spacing[0];
                xidx = ROUND_INT (x_in);
                if (zidx < 0 || zidx >= vol_in->dim[2] || yidx < 0 || yidx >= vol_in->dim[1] || xidx < 0 || xidx >= vol_in->dim[0]) {
                    for (d = 0; d < 3; d++) {
                        out_img[d][v] = 0.0;            /* Default value */
                    }
                } else {
                    for (d = 0; d < 3; d++) {
                        plm_long idx = zidx*vol_in->dim[1]*vol_in->dim[0] + yidx*vol_in->dim[0] + xidx;
                        out_img[d][v] = in_img[d][idx];
                    }
                }
            }
        }
    }

    return vol_out;
}

Volume*
volume_resample (Volume* vol_in, plm_long* dim, float* offset, float* spacing)
{
    switch (vol_in->pix_type) {
    case PT_UCHAR:
    case PT_SHORT:
    case PT_UINT32:
        fprintf (stderr, "Error, resampling PT_SHORT, PT_UCHAR, PT_UINT32 is unsupported\n");
        return 0;
    case PT_FLOAT:
        return volume_resample_float_li (vol_in, dim, offset, spacing);
    case PT_VF_FLOAT_INTERLEAVED:
        return volume_resample_vf_float_interleaved (vol_in, dim, offset, spacing);
    case PT_VF_FLOAT_PLANAR:
        return volume_resample_vf_float_planar (vol_in, dim, offset, spacing);
    case PT_UCHAR_VEC_INTERLEAVED:
        fprintf (stderr, "Error, resampling PT_UCHAR_VEC_INTERLEAVED is unsupported\n");
        return 0;
    default:
        fprintf (stderr, "Error, unknown pix_type: %d\n", vol_in->pix_type);
        return 0;
    }
}

Volume*
volume_resample_nn (Volume* vol_in, plm_long* dim, float* offset, float* spacing)
{
    switch (vol_in->pix_type) {
    case PT_UCHAR:
        volume_convert_to_float (vol_in);
        vol_in = volume_resample_float_nn (vol_in, dim, offset, spacing);
        volume_convert_to_uchar (vol_in);
        return vol_in;
    case PT_SHORT:
    case PT_UINT32:
        fprintf (stderr, "Error, resampling PT_SHORT and PT_UINT32 is unsupported\n");
        return 0;
    case PT_FLOAT:
        return volume_resample_float_nn (vol_in, dim, offset, spacing);
    case PT_VF_FLOAT_INTERLEAVED:
        return volume_resample_vf_float_interleaved (vol_in, dim, offset, spacing);
    case PT_VF_FLOAT_PLANAR:
        return volume_resample_vf_float_planar (vol_in, dim, offset, spacing);
    case PT_UCHAR_VEC_INTERLEAVED:
        fprintf (stderr, "Error, resampling PT_UCHAR_VEC_INTERLEAVED is unsupported\n");
        return 0;
    default:
        fprintf (stderr, "Error, unknown pix_type: %d\n", vol_in->pix_type);
        return 0;
    }
}

Volume*
volume_subsample (Volume* vol_in, int* sampling_rate)
{
    int d;
    plm_long dim[3];
    float offset[3];
    float spacing[3];

    for (d = 0; d < 3; d++) {
        float in_size = vol_in->dim[d] * vol_in->spacing[d];

        dim[d] = vol_in->dim[d] / sampling_rate[d];
        if (dim[d] < 1) dim[d] = 1;
        spacing[d] = in_size / dim[d];
        offset[d] = (float) (vol_in->offset[d] - 0.5 * vol_in->spacing[d] 
            + 0.5 * spacing[d]);
    }
    return volume_resample (vol_in, dim, offset, spacing);
}

Volume*
volume_subsample_nn (Volume* vol_in, int* sampling_rate)
{
    int d;
    plm_long dim[3];
    float offset[3];
    float spacing[3];

    for (d = 0; d < 3; d++) {
        float in_size = vol_in->dim[d] * vol_in->spacing[d];

        dim[d] = vol_in->dim[d] / sampling_rate[d];
        if (dim[d] < 1) dim[d] = 1;
        spacing[d] = in_size / dim[d];
        offset[d] = (float) (vol_in->offset[d] - 0.5 * vol_in->spacing[d] 
            + 0.5 * spacing[d]);
    }
    return volume_resample_nn (vol_in, dim, offset, spacing);
}
