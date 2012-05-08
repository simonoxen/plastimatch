/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_h_
#define _volume_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmbase.h"
 */

#include "plmbase_config.h"
#include "sys/plm_int.h"
#include "direction_cosines.h"
#include "volume_macros.h"

//TODO: Change type of directions_cosines to Direction_cosines*

class Volume_header;

enum Volume_pixel_type {
    PT_UNDEFINED,
    PT_UCHAR,
    PT_SHORT,
    PT_UINT16,
    PT_UINT32,
    PT_INT32,
    PT_FLOAT,
    PT_VF_FLOAT_INTERLEAVED,
    PT_VF_FLOAT_PLANAR,
    PT_UCHAR_VEC_INTERLEAVED
};


class API Volume
{
  public:
    plm_long dim[3];            // x, y, z Dims
    plm_long npix;              // # of voxels in volume
                                // = dim[0] * dim[1] * dim[2] 
    float offset[3];
    float spacing[3];
    Direction_cosines direction_cosines;
    float inverse_direction_cosines[9];
    float step[3][3];           // direction_cosines * spacing
    float proj[3][3];           // inv direction_cosines / spacing

    enum Volume_pixel_type pix_type;    // Voxel Data type
    int vox_planes;                     // # planes per voxel
    int pix_size;                       // # bytes per voxel
    void* img;                          // Voxel Data
  public:
    Volume ();
    Volume (
        const plm_long dim[3], 
        const float offset[3], 
        const float spacing[3], 
        const float direction_cosines[9], 
        enum Volume_pixel_type vox_type, 
        int vox_planes
    );
    Volume (
        const Volume_header& vh, 
        enum Volume_pixel_type vox_type, 
        int vox_planes
    );
    ~Volume ();
  public:
    void init () {
        for (int d = 0; d < 3; d++) {
            dim[d] = 0;
            offset[d] = 0;
            spacing[d] = 0;
        }
        for (int d = 0; d < 9; d++) {
            inverse_direction_cosines[d] = 0;
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                proj[i][j] = 0;
                step[i][j] = 0;
            }
        }
        npix = 0;
        pix_type = PT_UNDEFINED;
        vox_planes = 0;
        pix_size = 0;
        img = 0;
    }
    plm_long index (plm_long i, plm_long j, plm_long k) {
        return volume_index (this->dim, i, j, k);
    }
    void create (
        const plm_long dim[3], 
        const float offset[3], 
        const float spacing[3], 
        const float direction_cosines[9], 
        enum Volume_pixel_type vox_type, 
        int vox_planes = 1
    );
    void create (
        const Volume_header& vh, 
        enum Volume_pixel_type vox_type, 
        int vox_planes = 1
    );
    void set_direction_cosines (const float direction_cosines[9]);
  protected:
    void allocate (void);
};

C_API void vf_convert_to_interleaved (Volume* ref);
C_API void vf_convert_to_planar (Volume* ref, int min_size);
C_API void vf_pad_planar (Volume* vol, int size);  // deprecated?
C_API Volume* volume_clone_empty (Volume* ref);
C_API Volume* volume_clone (Volume* ref);
C_API void volume_convert_to_float (Volume* ref);
C_API void volume_convert_to_int32 (Volume* ref);
C_API void volume_convert_to_short (Volume* ref);
C_API void volume_convert_to_uchar (Volume* ref);
C_API void volume_convert_to_uint16 (Volume* ref);
C_API void volume_convert_to_uint32 (Volume* ref);
C_API Volume* volume_difference (Volume* vol, Volume* warped);
C_API Volume* volume_make_gradient (Volume* ref);
C_API void volume_matrix3x3inverse (float *out, const float *m);
C_API void volume_scale (Volume *vol, float scale);
C_API Volume* volume_warp (Volume* vout, Volume* vin, Volume* vf);
C_API void directions_cosine_debug (float *m);


#endif
