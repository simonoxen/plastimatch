/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "aperture.h"

class Aperture_private {
public:
    Aperture_private ()
    {
        ap_offset = 0.0;
        ires[0] = 0;
        ires[1] = 0;
    }
public:
    double ap_offset;
    int ires[2];
};

Aperture::Aperture ()
{
    this->d_ptr = new Aperture_private;

    this->vup[0] = 0.0;
    this->vup[1] = 0.0;
    this->vup[2] = 1.0;

    memset (this->ic,   0, 2*sizeof (double));
//    memset (this->ires, 0, 2*sizeof (int));
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
Aperture::get_offset () const
{
    return d_ptr->ap_offset;
}

void
Aperture::set_offset (double offset)
{
    d_ptr->ap_offset = offset;
}

const int*
Aperture::get_dim () const
{
    return d_ptr->ires;
}

int
Aperture::get_dim (int dim) const
{
    return d_ptr->ires[dim];
}

void
Aperture::set_dim (const int* dim)
{
    d_ptr->ires[0] = dim[0];
    d_ptr->ires[1] = dim[1];
}
