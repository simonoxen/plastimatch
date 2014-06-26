#include <stdio.h>

#include "plm_sleep.h"
#include "registration.h"
#include "rt_study.h"
#include "synthetic_mha.h"

int main ()
{
    Registration r;
    std::string s = 
        "[STAGE]\n"
        "xform=bspline\n"
        "max_its=15\n"
        "[STAGE]\n"
        "[STAGE]\n"
        ;
    r.set_command_string (s);

//    r.load_global_inputs ();

    Rt_study rtds;
    Synthetic_mha_parms sm_parms;
    sm_parms.pattern = PATTERN_RECT;
    synthetic_mha (&rtds, &sm_parms);
    Plm_image::Pointer fixed = rtds.get_image();
    sm_parms.pattern = PATTERN_SPHERE;
    synthetic_mha (&rtds, &sm_parms);
    Plm_image::Pointer moving = rtds.get_image();

    r.set_fixed_image (fixed);
    r.set_moving_image (moving);
    
    printf ("Calling start_registration\n");
    
    r.start_registration ();
    plm_sleep (1000);
    printf (">>> PAUSE\n");
    r.pause_registration ();
    plm_sleep (3000);
    printf (">>> PAUSE COMPLETE\n");
    r.start_registration ();
    r.wait_for_complete ();
    
    return 0;
}
