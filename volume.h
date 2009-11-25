/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_h_
#define _volume_h_

#include "plm_config.h"

enum Pixel_Type {
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

    enum Pixel_Type pix_type;	// Voxel Data type
    int pix_size;		// (Unused?)
    void* img;			// Voxel Data

#if defined (commentout)
    /* These are used for boundary testing */
    float xmin;	// Minimum X Value in Volume
    float xmax; // Maximum X Value in Volume
    float ymin; // Minimum Y Value in Volume
    float ymax; // Maximum Y Value in Volume
    float zmin; // Minimum Z Value in Volume
    float zmax; // Maximum Z Value in Volume
#endif
	float xmin;	// Minimum X Value in Volume
    float xmax; // Maximum X Value in Volume
    float ymin; // Minimum Y Value in Volume
    float ymax; // Maximum Y Value in Volume
    float zmin; // Minimum Z Value in Volume
    float zmax; // Maximum Z Value in Volume
};

#if defined __cplusplus
extern "C" {
#endif
int volume_index (int* dims, int k, int j, int i);
gpuit_EXPORT
Volume* volume_create (int* dim, float* offset, float* pix_spacing, 
		       enum Pixel_Type pix_type, float* direction_cosines, 
		       int min_size);
gpuit_EXPORT
void volume_free (Volume* vol);
gpuit_EXPORT
void volume_convert_to_float (Volume* ref);
void volume_convert_to_short (Volume* ref);
gpuit_EXPORT
void vf_convert_to_interleaved (Volume* ref);
void vf_convert_to_planar (Volume* ref, int min_size);
void vf_pad_planar (Volume* vol, int size);
Volume* volume_clone_empty (Volume* ref);
Volume* volume_clone (Volume* ref);
gpuit_EXPORT
Volume* volume_make_gradient (Volume* ref);
Volume* warp_image (Volume* vol, float** vec);
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
