/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_timer_h_
#define _plm_timer_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmsys.h"
 */

#include "plmsys_config.h"

class Plm_timer_private;

class API Plm_timer {
public:
    Plm_timer ();
    ~Plm_timer ();

    void start ();
    double report ();
private:
    Plm_timer_private *d_ptr;
};

#endif
