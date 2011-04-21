/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _to_string_h_
#define _to_string_h_

#include "plm_config.h"
#include <iomanip>
#include <sstream>

template <class T>
static std::string 
to_string (
    T t, 
    int width = 0, 
    char fill = ' ',
    std::ios_base & (*f)(std::ios_base&) = std::dec
)
{
   std::stringstream ss;
   ss << std::setw (width);
   ss << std::setfill (fill);
   ss << f << t;
   return ss.str();
}

#endif
