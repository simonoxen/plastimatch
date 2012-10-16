/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proj_volume_h_
#define _proj_volume_h_

#include "plmbase_config.h"

class Proj_matrix;
class Proj_volume_private;
class Volume;

/*! \brief 
 * The Proj_volume class represents a three-dimensional volume 
 * on a uniform non-orthogonal grid.  The grid is regular within 
 * a rectangular frustum, the geometry of which is specified by 
 * a projection matrix.
 */
class Proj_volume 
{
public:
    Proj_volume ();
    ~Proj_volume ();
public:
    Proj_volume_private *d_ptr;
};

#endif
