/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_boundary_behavior_h_
#define _volume_boundary_behavior_h_

#include "plmbase_config.h"

/*! \brief This enum is used to control the algorithm behavior 
  for voxels at the edge of the volume.  If ZERO_PADDING is 
  specified, all non-zero voxels at the edge of the volume 
  will be treated as boundary voxels.  If EDGE_PADDING is 
  specified, non-zero voxels at the edge of the volume are 
  only treated as boundary voxels if they neighbor 
  a zero voxel.  If ADAPTIVE_PADDING, it will use 
  EDGE_PADDING for dimensions of a single voxel, and ZERO_PADDING 
  for dimensions of multiple voxels. */
enum Volume_boundary_behavior {
    ZERO_PADDING,
    EDGE_PADDING,
    ADAPTIVE_PADDING
};

const unsigned char VBB_MASK_NEG_I = 0x01;
const unsigned char VBB_MASK_NEG_J = 0x02;
const unsigned char VBB_MASK_NEG_K = 0x04;
const unsigned char VBB_MASK_POS_I = 0x08;
const unsigned char VBB_MASK_POS_J = 0x10;
const unsigned char VBB_MASK_POS_K = 0x20;

PLMBASE_API Volume_boundary_behavior
volume_boundary_behavior_parse (const std::string& string);
PLMBASE_API Volume_boundary_behavior
volume_boundary_behavior_parse (const char* string);

#endif
