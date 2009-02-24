/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_h_
#define _volume_h_

enum Pixel_Type {
    PT_UNDEFINED,
    PT_UCHAR,
    PT_SHORT,
    PT_FLOAT,
    PT_VF_FLOAT_INTERLEAVED,
    PT_VF_FLOAT_PLANAR
};

typedef struct volume Volume;
struct volume
{
    int dim[3];		// x, y, z Dims
    int npix;		// # of pixels in volume
			// = dim[0] * dim[1] * dim[2] 
    float offset[3];
    float pix_spacing[3];	// voxel spacing

    enum Pixel_Type pix_type;	// Voxel Data type
    int pix_size;		// (Unused??)
    void* img;			// Voxel Data

    /* These are used for boundary testing */
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
Volume* volume_create (int* dim, float* offset, float* pix_spacing, 
		       enum Pixel_Type pix_type, int min_size);
void volume_free (Volume* vol);
void volume_convert_to_float (Volume* ref);
void volume_convert_to_short (Volume* ref);
void vf_convert_to_interleaved (Volume* ref);
void vf_convert_to_planar (Volume* ref, int min_size);
void vf_pad_planar (Volume* vol, int size);
Volume* volume_clone_empty (Volume* ref);
Volume* volume_clone (Volume* ref);
Volume* volume_make_gradient (Volume* ref);
Volume* warp_image (Volume* vol, float** vec);
Volume* volume_difference (Volume* vol, Volume* warped);
Volume* volume_warp (Volume* vout, Volume* vin, Volume* vf);
Volume* volume_resample (Volume* vol_in, int* dim, float* offset, float* pix_spacing);
Volume* volume_subsample (Volume* vol_in, int* sampling_rate);
void vf_print_stats (Volume* vol);
void vf_convolve_x (Volume* vf_out, Volume* vf_in, float* ker, int width);
void vf_convolve_y (Volume* vf_out, Volume* vf_in, float* ker, int width);
void vf_convolve_z (Volume* vf_out, Volume* vf_in, float* ker, int width);
#if defined __cplusplus
}
#endif

#endif
