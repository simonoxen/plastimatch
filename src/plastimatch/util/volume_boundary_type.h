/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_boundary_type_h_
#define _volume_boundary_type_h_

#include "plmbase_config.h"

/*! \brief This enum is used to control the output type for 
  volume bounary detection.  If INTERIOR_EDGE_VOXELS, the 
  boundary will be the set of voxels within the image region that 
  lie on the border of the region.  If FACE_EDGE_AND_CORNER, 
  the output image has dimension one larger than the input image, 
  and each voxel is a bitmask of size seven, which correspond to a 
  boundary at the voxel corner where all three dimensions have 
  negative index, the boundary at three edges where two dimensions 
  have negative index, and the boundary at three faces where 
  the face has negative index.
  Note however, that corner and edges on the volume boundary 
  are not guaranteed to be correct, since these are not used 
  by the downstream distance map code. */
enum Volume_boundary_type {
    INTERIOR_EDGE_VOXELS,
    FACE_EDGE_AND_CORNER
};

#endif
