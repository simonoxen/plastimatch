/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rpl_volume_h_
#define _rpl_volume_h_

#include "plmbase_config.h"
#include <string>

class Proj_volume;
class Ray_data;
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
    Proj_volume *get_proj_volume ();
    double get_rgdepth (const double *xyz);
    void save (const std::string& filename);
    void save (const char* filename);

    void compute_wed_volume (Volume *wed_vol, Volume *in_vol, float background);
    void compute_dew_volume (Volume *wed_vol, Volume *dew_vol, float background);
    void compute_segdepth_volume (Volume *seg_vol, Volume *aperture_vol, Volume *segdepth_vol, float background);

protected:
    void ray_trace (
        Volume *ct_vol,              /* I: CT volume */
        Ray_data *ray_data,          /* I: Pre-computed data for this ray */
        Volume_limit *vol_limit,     /* I: CT bounding region */
        const double *src,           /* I: @ source */
        int* ires                    /* I: ray cast resolution */
    );
};

#endif
