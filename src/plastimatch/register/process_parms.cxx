/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "shared_parms.h"
#include "process_parms.h"

class Process_parms_private
{
public:
    Shared_parms *shared;
public:
    Process_parms_private () {
        shared = new Shared_parms;
    }
    Process_parms_private (const Process_parms_private& s) {
        this->shared = new Shared_parms (*s.shared);
    }
    ~Process_parms_private () {
        delete shared;
    }
};

Process_parms::Process_parms ()
{
    d_ptr = new Process_parms_private;

}

Process_parms::Process_parms (const Process_parms& s) 
{
    d_ptr = new Process_parms_private (*s.d_ptr);
}

Process_parms::~Process_parms ()
{
    delete d_ptr;
}
