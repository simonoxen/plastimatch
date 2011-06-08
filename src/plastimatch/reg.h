/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _reg_h_
#define _reg_h_

#include "plm_config.h"
#include "volume.h"

gpuit_EXPORT
float
vf_regularize_numerical (Volume* vol);

class Reg_parms
{
public:
    char implementation;    /* Implementation: a, b, c, etc */
public:
    Reg_parms () {
        /* Init */
        this->implementation = '\0';
    }
};

#endif /* _reg_h_ */
