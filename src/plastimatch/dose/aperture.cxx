/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "plmdose.h"

Aperture::Aperture ()
{
    this->ap_offset = 0.0;

    this->vup[0] = 0.0;
    this->vup[1] = 0.0;
    this->vup[2] = 1.0;

    memset (this->ic,   0, 2*sizeof (double));
    memset (this->ires, 0, 2*sizeof (int));
    memset (this->ic_room, 0, 3*sizeof (double));
    memset (this->ul_room, 0, 3*sizeof (double));
    memset (this->incr_r, 0, 3*sizeof (double));
    memset (this->incr_c, 0, 3*sizeof (double));
}
