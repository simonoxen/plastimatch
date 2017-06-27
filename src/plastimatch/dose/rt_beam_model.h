/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rt_beam_model_h_
#define _rt_beam_model_h_

#include "plmdose_config.h"
#include "smart_pointer.h"

class Rt_beam_model_private;

class PLMDOSE_API Rt_beam_model {
public:
    SMART_POINTER_SUPPORT (Rt_beam_model);
    Rt_beam_model_private *d_ptr;
public:
    Rt_beam_model ();
    ~Rt_beam_model ();
};

#endif
