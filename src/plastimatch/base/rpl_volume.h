/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rpl_volume_h_
#define _rpl_volume_h_

#include "plmbase_config.h"
#include <string>
#include "aperture.h"
#include "plm_image.h"
#include "ray_trace_callback.h"

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

    void set_ct_volume (Plm_image::Pointer& ct_volume);

    Aperture::Pointer& get_aperture ();
    const Aperture::Pointer& get_aperture () const;
    void set_aperture (Aperture::Pointer& ap);

    Volume* get_vol ();
    const Volume* get_vol () const;
    Proj_volume* get_proj_volume ();

    double get_rgdepth (int ap_ij[2], double dist);
    double get_rgdepth (double ap_ij[2], double dist);
    double get_rgdepth (const double *xyz);
    double get_rgdepth2 (const double *xyz);

    void set_ct (const Plm_image::Pointer& ct_volume);
    Plm_image::Pointer get_ct();
    void set_ct_limit(Volume_limit* ct_limit);
    Volume_limit* get_ct_limit();
    void set_ray(Ray_data *ray);
    Ray_data* get_Ray_data();
    void set_front_clipping_plane(double front_clip);
    double get_front_clipping_plane () const;
    void set_back_clipping_plane(double back_clip);
    double get_back_clipping_plane () const;

    double get_max_wed ();
    double get_min_wed ();

    void compute_rpl ();
    void compute_rpl_RSP();
    void compute_rpl_ct ();
    void compute_rpl_void ();
    void compute_rpl_rglength_wo_rg_compensator ();
    void compute (Volume *ct_vol);

    double compute_farthest_penetrating_ray_on_nrm(float range); // return the distance from aperture to the farthest which rg_lenght > range

    void compute_wed_volume (Volume *wed_vol, Volume *in_vol, float background);
    void compute_dew_volume (Volume *wed_vol, Volume *dew_vol, float background);
    Volume* create_proj_wed_volume ();
    void compute_proj_wed_volume (Volume *proj_wed_vol, float background);
    void compute_beam_modifiers (Volume *seg_vol, float background);
    void compute_aperture (Volume *tgt_vol, float background);

    void apply_beam_modifiers ();

    void save (const std::string& filename);
    void save (const char* filename);

    void compute_ray_data ();

protected:

    void aprc_ray_trace (
        Volume *tgt_vol,             /* I: CT volume */
        Ray_data *ray_data,          /* I: Pre-computed data for this ray */
        Volume_limit *vol_limit,     /* I: CT bounding region */
        const double *src,           /* I: @ source */
        double rc_thk,               /* I: range compensator thickness */
        int* ires                    /* I: ray cast resolution */
    );
    void rpl_ray_trace (
        Volume *ct_vol,              /* I: CT volume */
        Ray_data *ray_data,          /* I: Pre-computed data for this ray */
        Ray_trace_callback callback, /* I: Ray trace callback function */
        Volume_limit *vol_limit,     /* I: CT bounding region */
        const double *src,           /* I: @ source */
        double rc_thk,               /* I: range compensator thickness */
        int* ires                    /* I: ray cast resolution */
    );

};

#endif
