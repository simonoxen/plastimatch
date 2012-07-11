/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "fatm.h"


void
tracker_initialize ()
{
#if defined (this_is_today)
    FATM_Options *fopt;
    int pat_size[] = { 21, 21 };
    int sig_size[] = { 151, 151 };

    /* Initialize fopt array */
    fopt = fatm_initialize ();

    /* 
    fopt->pat_rect = pointer to pattern
    fopt->sig_rect = pointer to signal 
    fopt->score = pointer to score
    */

    fopt->command = MATCH_COMMAND_COMPILE;
    fopt->alg = MATCH_ALGORITHM_FNCC;
    fatm_compile (fopt);

    fatm_run (fopt);
    /*
    fopt->pat_rect = new pattern 
    fatm_run (fopt);
    */
#endif

#if defined (this_is_what_we_would_like)
    Fatm fatm = new Fatm;
    Image_rect pattern;
    pattern->set_size (21, 21);
    unsigned short *us = pattern->get_image_pointer();

    for (int i = 0; i < 21; i++) {
        for (int j = 0; j < 21; j++) {
            if (i*j < 33) {
                *us = -1;
            } else {
                *us = 1;
            }
        }
    }

    fatm->set_pattern (pattern);
#endif
}
