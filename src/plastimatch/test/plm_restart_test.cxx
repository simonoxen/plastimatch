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
#if defined (commentout)
        "max_its=1\n"
#endif
        "max_its=100\n"
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
    
#if defined (commentout)
    r.load_global_inputs ();
//    r.load_global_inputs ();
#endif

    r.set_fixed_image (fixed);
    r.set_moving_image (moving);

    printf ("Calling start_registration\n");

    r.start_registration ();
    plm_sleep (1000);
    printf (">>> PAUSE\n");
    r.pause_registration ();
    printf (">>> PAUSE RETURNED\n");
    plm_sleep (3000);
    printf (">>> PAUSE COMPLETE\n");
    r.start_registration ();
    r.wait_for_complete ();

#if defined (commentout)    
    r.do_registration_pure_old ();
#endif

    return 0;
}
