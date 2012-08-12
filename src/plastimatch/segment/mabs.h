/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mabs_h_
#define _mabs_h_

#include "plmsegment_config.h"
#include <string>

class Mabs_private;
class Mabs_parms;

class PLMSEGMENT_API Mabs {
public:
    Mabs ();
    ~Mabs ();
public:
    Mabs_private *d_ptr;

public:
    std::string map_structure_name (
        const Mabs_parms& parms, 
        const std::string& ori_name);
    void
    run (const Mabs_parms& parms);
};

#endif
