/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_h_
#define _volume_h_

#include "plm_config.h"

enum Volume_pixel_type {
    PT_UNDEFINED,
    PT_UCHAR,
    PT_SHORT,
    PT_UINT32,
    PT_FLOAT,
    PT_VF_FLOAT_INTERLEAVED,
    PT_VF_FLOAT_PLANAR
};

typedef struct volume Volume;
struct volume
{
    int dim[3];			// x, y, z Dims
    int npix;			// # of pixels in volume
				// = dim[0] * dim[1] * dim[2] 
    float offset[3];
    float pix_spacing[3];	// voxel spacing
    float direction_cosines[9];

    enum Volume_pixel_type pix_type;	// Voxel Data type
    int pix_size;		// (Unused?)
    void* img;			// Voxel Data
};

#if defined __cplusplus
extern "C" {
#endif
gpuit_EXPORT
int volume_index (int* dims, int i, int j, int k);
gpuit_EXPORT
Volume*
volume_create (
    int dim[3], 
    float offset[3], 
    float pix_spacing[3], 
    enum Volume_pixel_type pix_type, 
    float direction_cosines[9],
    int min_size
);
gpuit_EXPORT
void volume_destroy (Volume* vol);
gpuit_EXPORT
void volume_convert_to_float (Volume* ref);
gpuit_EXPORT
void volume_convert_to_short (Volume* ref);
gpuit_EXPORT
void
volume_convert_to_uint32 (Volume* ref);
gpuit_EXPORT
void vf_convert_to_interleaved (Volume* ref);
void vf_convert_to_planar (Volume* ref, int min_size);
void vf_pad_planar (Volume* vol, int size);
gpuit_EXPORT
Volume* volume_clone_empty (Volume* ref);
gpuit_EXPORT
Volume* volume_clone (Volume* ref);
gpuit_EXPORT
Volume* volume_make_gradient (Volume* ref);
Volume* volume_difference (Volume* vol, Volume* warped);
gpuit_EXPORT
Volume* volume_warp (Volume* vout, Volume* vin, Volume* vf);
gpuit_EXPORT
Volume* volume_resample (Volume* vol_in, int* dim, float* offset, float* pix_spacing);
gpuit_EXPORT
Volume* volume_subsample (Volume* vol_in, int* sampling_rate);
gpuit_EXPORT
void vf_print_stats (Volume* vol);
gpuit_EXPORT
void vf_convolve_x (Volume* vf_out, Volume* vf_in, float* ker, int width);
gpuit_EXPORT
void vf_convolve_y (Volume* vf_out, Volume* vf_in, float* ker, int width);
gpuit_EXPORT
void vf_convolve_z (Volume* vf_out, Volume* vf_in, float* ker, int width);
gpuit_EXPORT
Volume* 
volume_axial2coronal (Volume* ref);
gpuit_EXPORT
Volume* 
volume_axial2sagittal (Volume* ref);

#if defined __cplusplus
}
#endif

#endif
