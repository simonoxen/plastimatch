#include <stdio.h>

#include "plm_sleep.h"
#include "registration.h"
#include "rt_study.h"
#include "synthetic_mha.h"

int main ()
{
    Registration r;
    std::string s = 
        "[GLOBAL]\n"
        "fixed=c:/tmp/fixed.mha\n"
        "moving=c:/tmp/moving.mha\n"
        "[STAGE]\n"
        "xform=bspline\n"
        "max_its=1\n"
        "[STAGE]\n"
        ;
    r.set_command_string (s);

    Plm_image::Pointer fixed;
    {
    Rt_study rtds;
    Synthetic_mha_parms sm_parms;
    sm_parms.pattern = PATTERN_RECT;
    synthetic_mha (&rtds, &sm_parms);
    fixed = rtds.get_image();
    }
    Plm_image::Pointer moving;
    {
    Rt_study rtds;
    Synthetic_mha_parms sm_parms;
    sm_parms.pattern = PATTERN_SPHERE;
    synthetic_mha (&rtds, &sm_parms);
    moving = rtds.get_image();
    }
    
    r.load_global_inputs ();

    r.set_fixed_image (fixed);
    r.set_moving_image (moving);
#if defined (commentout)
#endif

    printf ("Calling start_registration\n");

#if defined (commentout)    
    r.start_registration ();
    plm_sleep (1000);
    printf (">>> PAUSE\n");
    r.pause_registration ();
    plm_sleep (3000);
    printf (">>> PAUSE COMPLETE\n");
    r.start_registration ();
    r.wait_for_complete ();
#endif

    r.do_registration_pure_old ();

    return 0;
}
