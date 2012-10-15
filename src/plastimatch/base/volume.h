/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_h_
#define _volume_h_

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

/*! \brief 
 * The Volume class represents a three-dimensional volume on a uniform 
 * grid.  The volume can be located at arbitrary positions and orientations 
 * in space, and can represent most voxel types (float, unsigned char, etc.).
 * A volume can also support multiple planes, which is used to hold 
 * three dimensional vector fields, or three-dimensional bitfields.  
 */
class PLMBASE_API Volume
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
    /*! \brief Make a copy of the volume */
    Volume* clone ();
    /*! \brief Convert the image voxels to a new data type */
    void convert (Volume_pixel_type new_type);

    /*! \brief Get a pointer to the direction cosines.  
      Direction cosines hold the orientation of a volume. 
      They are defined as the unit length direction vectors 
      of the volume in world space as one traverses the pixels
      in the raw array of values.
    */
    float* get_direction_cosines (void);
    /*! \brief Set the direction cosines.  
      Direction cosines hold the orientation of a volume. 
      They are defined as the unit length direction vectors 
      of the volume in world space as one traverses the pixels
      in the raw array of values.
    */
    void set_direction_cosines (const float direction_cosines[9]);
  protected:
    void allocate (void);
    void init ();
};

PLMBASE_C_API void vf_convert_to_interleaved (Volume* ref);
PLMBASE_C_API void vf_convert_to_planar (Volume* ref, int min_size);
PLMBASE_C_API void vf_pad_planar (Volume* vol, int size);  // deprecated?
PLMBASE_C_API Volume* volume_clone_empty (Volume* ref);
PLMBASE_C_API Volume* volume_clone (Volume* ref);
PLMBASE_C_API void volume_convert_to_float (Volume* ref);
PLMBASE_C_API void volume_convert_to_int32 (Volume* ref);
PLMBASE_C_API void volume_convert_to_short (Volume* ref);
PLMBASE_C_API void volume_convert_to_uchar (Volume* ref);
PLMBASE_C_API void volume_convert_to_uint16 (Volume* ref);
PLMBASE_C_API void volume_convert_to_uint32 (Volume* ref);
PLMBASE_C_API Volume* volume_difference (Volume* vol, Volume* warped);
PLMBASE_C_API Volume* volume_make_gradient (Volume* ref);
PLMBASE_C_API void volume_matrix3x3inverse (float *out, const float *m);
PLMBASE_C_API void volume_scale (Volume *vol, float scale);
PLMBASE_C_API Volume* volume_warp (Volume* vout, Volume* vin, Volume* vf);
PLMBASE_C_API void directions_cosine_debug (float *m);


#endif
