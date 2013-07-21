/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "aperture.h"
#include "plm_image.h"

class Aperture_private {
public:
    Aperture_private ()
    {
        distance = 0.0;
        dim[0] = 0;
        dim[1] = 0;
        center[0] = 0;
        center[1] = 0;
        spacing[0] = 0;
        spacing[1] = 0;
    }
public:
    Plm_image::Pointer aperture_image;
    Plm_image::Pointer range_compensator_image;

    double distance;
    int dim[2];
    double center[2];
    double spacing[2];
};

Aperture::Aperture ()
{
    this->d_ptr = new Aperture_private;

    this->vup[0] = 0.0;
    this->vup[1] = 0.0;
    this->vup[2] = 1.0;

    //memset (this->ic,   0, 2*sizeof (double));
    memset (this->ic_room, 0, 3*sizeof (double));
    memset (this->ul_room, 0, 3*sizeof (double));
    memset (this->incr_r, 0, 3*sizeof (double));
    memset (this->incr_c, 0, 3*sizeof (double));
}

Aperture::~Aperture ()
{
    delete this->d_ptr;
}

double
Aperture::get_distance () const
{
    return d_ptr->distance;
}

void
Aperture::set_distance (double distance)
{
    d_ptr->distance = distance;
}

const int*
Aperture::get_dim () const
{
    return d_ptr->dim;
}

int
Aperture::get_dim (int dim) const
{
    return d_ptr->dim[dim];
}

void
Aperture::set_dim (const int* dim)
{
    d_ptr->dim[0] = dim[0];
    d_ptr->dim[1] = dim[1];
    d_ptr->center[0] = (dim[0]-1) / 2;
    d_ptr->center[1] = (dim[1]-1) / 2;
}

const double*
Aperture::get_center () const
{
    return d_ptr->center;
}

double
Aperture::get_center (int dim) const
{
    return d_ptr->center[dim];
}

void
Aperture::set_center (const float* center)
{
    d_ptr->center[0] = center[0];
    d_ptr->center[1] = center[1];
}

void
Aperture::set_center (const double* center)
{
    d_ptr->center[0] = center[0];
    d_ptr->center[1] = center[1];
}

void
Aperture::set_origin (const float* origin)
{
    /* GCS FIX: This should be saved internally as an origin, 
       then only converted to pixel coordinates as needed */

    /* Compute & save as pixel coordinates */
    for (int i = 0; i < 2; i++) {
        d_ptr->center[i] = ((double) -origin[i]) / d_ptr->spacing[i];
    }
}

const double*
Aperture::get_spacing () const
{
    return d_ptr->spacing;
}

double
Aperture::get_spacing (int dim) const
{
    return d_ptr->spacing[dim];
}

void
Aperture::set_spacing (const float* spacing)
{
    d_ptr->spacing[0] = spacing[0];
    d_ptr->spacing[1] = spacing[1];
}

void
Aperture::set_spacing (const double* spacing)
{
    d_ptr->spacing[0] = spacing[0];
    d_ptr->spacing[1] = spacing[1];
}

void
Aperture::allocate_aperture_images ()
{
    plm_long dim[3] = {
        d_ptr->dim[0],
        d_ptr->dim[1],
        1
    };
    float origin[3] = { 0, 0, 0 };
    float spacing[3];
    spacing[0] = d_ptr->spacing[0];
    spacing[1] = d_ptr->spacing[1];
    spacing[2] = 1;

    Volume *ap_vol = new Volume (dim, origin, spacing, NULL, PT_UCHAR, 1);
    Volume *rc_vol = new Volume (dim, origin, spacing, NULL, PT_FLOAT, 1);

    d_ptr->aperture_image = Plm_image::New (new Plm_image (ap_vol));
    d_ptr->range_compensator_image = Plm_image::New (new Plm_image (rc_vol));
}

bool
Aperture::have_aperture_image ()
{
    return (bool) d_ptr->aperture_image;
}

Plm_image::Pointer&
Aperture::get_aperture_image ()
{
    return d_ptr->aperture_image;
}

Volume*
Aperture::get_aperture_volume ()
{
    if (!d_ptr->aperture_image) {
        return 0;
    }
    return d_ptr->aperture_image->get_vol_uchar();
}

void 
Aperture::set_aperture_image (const char *ap_filename)
{
    d_ptr->aperture_image = Plm_image::New (new Plm_image(ap_filename));
}

bool
Aperture::have_range_compensator_image ()
{
    return (bool) d_ptr->range_compensator_image;
}

Plm_image::Pointer&
Aperture::get_range_compensator_image ()
{
    return d_ptr->range_compensator_image;
}

Volume*
Aperture::get_range_compensator_volume ()
{
    if (!d_ptr->range_compensator_image) {
        return 0;
    }
    return d_ptr->range_compensator_image->get_vol_float();
}

void 
Aperture::set_range_compensator_image (const char *rc_filename)
{
    d_ptr->range_compensator_image 
        = Plm_image::New (new Plm_image(rc_filename));
}
