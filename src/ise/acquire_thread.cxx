/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include "acquire_thread.h"
#include "dips_panel.h"
#include "varian_4030e.h"

Acquire_thread::Acquire_thread()
{
    this->dp = new Dips_panel;
}

Acquire_thread::~Acquire_thread()
{
    if (this->dp) {
        delete this->dp;
    }
    if (this->vp) {
        delete this->vp;
    }
}

void 
Acquire_thread::open_receptor (const char* path)
{
    int result;
    this->vp = new Varian_4030e (this->idx);
    result = vp->open_link (path);
    result = vp->disable_missing_corrections (result);
    if (result != HCP_NO_ERR) {
	printf ("vp.open_receptor_link returns error (%d): %s\n", result, 
	    Varian_4030e::error_string(result));
        return;
        //return -1;
    }
    result = vp->check_link ();
    if (result != HCP_NO_ERR) {
	printf ("vp.check_link returns error %d\n", result);
        vip_close_link();
        return;
        //return -1;
    }
    vp->print_sys_info ();

    result = vip_select_mode (vp->current_mode);
    vp->print_mode_info ();

    if (result != HCP_NO_ERR) {
        printf ("vip_select_mode(%d) returns error %d\n", 
            vp->current_mode, result);
        vp->close_link ();
        //return -1;
    }
}

void 
Acquire_thread::run()
{
    dp->open_panel (this->idx, HIRES_IMAGE_HEIGHT, HIRES_IMAGE_WIDTH);

    while (true) {
#if defined (commentout)
        /* Wait for generator expose request */
        while (!advantech.ready_for_expose ()) {
            Sleep (10);
        }
#endif
        /* Wait for, and save frame from panel */
        vp->rad_acquisition (dp);
    }
    vip_close_link();
}
