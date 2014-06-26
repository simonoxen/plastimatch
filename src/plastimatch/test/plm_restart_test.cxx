#include <stdio.h>
#include <windows.h>
#include "registration.h"

int main ()
{
    Registration r;
    std::string s = 
        "[GLOBAL]\n"
        "fixed=C:/tmp/fixed.mha\n"
        "moving=C:/tmp/moving.mha\n"
        "[STAGE]\n"
        "xform=bspline\n"
        "max_its=90\n"
        "[STAGE]\n"
        "[STAGE]\n";
    
    r.set_command_string (s);

    r.load_global_inputs ();
    printf ("Calling start_registration\n");
    
    r.start_registration ();
    Sleep (1000);
    printf (">>> PAUSE\n");
    r.pause_registration ();
    Sleep (3000);
    printf (">>> PAUSE COMPLETE\n");
    r.start_registration ();
    r.wait_for_complete ();
    
    return 0;
}
