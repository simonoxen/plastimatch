/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _synthetic_source_h_
#define _synthetic_source_h_

#include "fluoro_source.h"

class Synthetic_source : public Fluoro_source {
public:
    Synthetic_source ();
public:
    virtual unsigned long get_size_x (void);
    virtual unsigned long get_size_y (void);
    virtual const std::string get_type (void);
};

void synthetic_grab_image (Frame* f);

#endif
