/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_h_
#define _volume_h_

#include "plmbase_config.h"

#include "plmsys.h"

#include "volume_header.h"
#include "volume_macros.h"


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


class gpuit_EXPORT Volume
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
    Volume () {
        init ();
    }
    Volume (
        const plm_long dim[3], 
        const float offset[3], 
        const float spacing[3], 
        const float direction_cosines[9], 
        enum Volume_pixel_type vox_type, 
        int vox_planes
    ) {
        init ();
        create (dim, offset, spacing, direction_cosines, vox_type, 
            vox_planes);
    }
    Volume (
        const Volume_header& vh, 
        enum Volume_pixel_type vox_type, 
        int vox_planes
    ) {
        init ();
        create (vh, vox_type, vox_planes);
    }
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

#endif
