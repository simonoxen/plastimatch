/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rpl_volume_h_
#define _rpl_volume_h_

#include "plmbase_config.h"

class Rpl_volume_private;
class Volume;
class Volume_limit;

class PLMBASE_API Rpl_volume 
{
public:
    Rpl_volume ();
    ~Rpl_volume ();
public:
    Rpl_volume_private *d_ptr;
public:
    void set_geometry (
        const double src[3],           // position of source (mm)
        const double iso[3],           // position of isocenter (mm)
        const double vup[3],           // dir to "top" of projection plane
        double sid,                    // dist from proj plane to source (mm)
        const int image_dim[2],        // resolution of image
        const double image_center[2],  // image center (pixels)
        const double image_spacing[2], // pixel size (mm)
        const double step_length       // spacing between planes
    );

    void compute (Volume *ct_vol);

    Volume* get_volume ();
    double get_rgdepth (const double *xyz);
    void save (const char* filename);
protected:
    void ray_trace (
        Volume *ct_vol,              /* I: CT volume */
        Volume_limit *vol_limit,     /* I: CT bounding region */
        const double *p1,            /* I: @ source */
        const double *p2,            /* I: @ aperture */
        int* ires,                   /* I: ray cast resolution */
        int ap_idx                   /* I: linear index of ray in ap */
    );

};

#endif
