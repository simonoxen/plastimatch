/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _process_parms_h_
#define _process_parms_h_

#include "plmregister_config.h"
#include <list>
#include <string>
#include <ctype.h>
#include <stdlib.h>

class Plm_image;
class Shared_parms;
class Process_parms_private;

class PLMREGISTER_API Process_parms {
public:
    Process_parms_private *d_ptr;
public:
    Process_parms ();
    Process_parms (const Process_parms& s);
    ~Process_parms ();
};

#endif
