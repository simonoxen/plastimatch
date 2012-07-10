/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "fatm.h"


void
tracker_initialize ()
{
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
}
