/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_h_
#define _volume_h_

#include "plmbase_config.h"

#include "direction_cosines.h"
#include "plm_int.h"
#include "smart_pointer.h"
#include "volume_macros.h"

//TODO: Change type of directions_cosines to Direction_cosines*

class Volume_header;

enum Volume_pixel_type {
    PT_UNDEFINED,
    PT_UCHAR,
    PT_UINT16,
    PT_SHORT,
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
    SMART_POINTER_SUPPORT (Volume);
public:
    plm_long dim[3];            // x, y, z Dims
    plm_long npix;              // # of voxels in volume
                                // = dim[0] * dim[1] * dim[2] 
    float offset[3];
    float spacing[3];
    Direction_cosines direction_cosines;

    enum Volume_pixel_type pix_type;    // Voxel Data type
    int vox_planes;                     // # planes per voxel
    int pix_size;                       // # bytes per voxel
    void* img;                          // Voxel Data

public:
    float step[9];           // direction_cosines * spacing
    float proj[9];           // inv direction_cosines / spacing

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
        const plm_long new_dim[3], 
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
    Volume::Pointer clone ();

    /*! \brief Make a copy of the volume with the same geometry 
      and same voxel values, but converted to a 
      different data type */
    Volume::Pointer clone (Volume_pixel_type new_type) const;

    /*! \brief Make a copy of the volume with the same geometry, 
      but set all the voxel values to zero. */
    Volume::Pointer clone_empty ();

    /*! \brief Convert the image voxels to a new data type */
    void convert (Volume_pixel_type new_type);

    /*! \brief Get a pointer to the origin of the volume.  
      The origin is defined as the location in world coordinates 
      of the center of the first voxel in the volume.
    */
    const float* get_origin (void);
    /*! \brief Set the origin.
      The origin is defined as the location in world coordinates 
      of the center of the first voxel in the volume.
    */
    void set_origin (const float origin[3]);
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

    /*! \brief Get the raw image pointer as specified type.
      No error checking done.
    */
    template<class T> T* get_raw ();
    template<class T> const T* get_raw () const;

    /*! \brief Get the step matrix.
      The step matrix encodes the transform from voxel coordinates 
      to world coordinates.
    */
    const float* get_step (void) const;
    /*! \brief Get the proj matrix.
      The proj matrix encodes the transform from world coordinates
      to voxel coordinates.
    */
    const float* get_proj (void) const;

    /*! \brief Get the value at a voxel coordinate, clamped and 
      tri-linearly interpolated.  Only applies to float volumes.
    */
    float get_ijk_value (const float xyz[3]);
    void get_xyz_from_ijk (double xyz[3], const int ijk[3]);
    void get_ijk_from_xyz (int ijk[3],const float xyz[3], bool* in);
	void get_ijk_from_xyz (float ijk[3],const float xyz[3], bool* in);

    /*! \brief In-place (destructive) scaling of the image according to the 
      supplied scale factor */
    void scale_inplace (float scale);

    void debug ();
    void direction_cosines_debug ();
protected:
    void allocate (void);
    void init ();
public:
    /* Some day, these should become protected */
    Volume* clone_raw ();
};

PLMBASE_C_API void vf_convert_to_interleaved (Volume* ref);
PLMBASE_C_API void vf_convert_to_planar (Volume* ref, int min_size);
PLMBASE_C_API void vf_pad_planar (Volume* vol, int size);  // deprecated?
PLMBASE_C_API Volume* volume_clone_empty (Volume* ref);
PLMBASE_C_API Volume* volume_clone (const Volume* ref);
PLMBASE_C_API void volume_convert_to_float (Volume* ref);
PLMBASE_C_API void volume_convert_to_int32 (Volume* ref);
PLMBASE_C_API void volume_convert_to_short (Volume* ref);
PLMBASE_C_API void volume_convert_to_uchar (Volume* ref);
PLMBASE_C_API void volume_convert_to_uint16 (Volume* ref);
PLMBASE_C_API void volume_convert_to_uint32 (Volume* ref);
PLMBASE_C_API Volume* volume_difference (Volume* vol, Volume* warped);
PLMBASE_C_API Volume* volume_make_gradient (Volume* ref);
PLMBASE_C_API void volume_matrix3x3inverse (float *out, const float *m);

#endif
