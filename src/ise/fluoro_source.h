/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fluoro_source_h_
#define _fluoro_source_h_

#include <string>

class Fluoro_source {
public:
    Fluoro_source ();
public:
    virtual unsigned long get_size_x (void) = 0;
    virtual unsigned long get_size_y (void) = 0;
    virtual const std::string get_type (void) = 0;
};

#endif
