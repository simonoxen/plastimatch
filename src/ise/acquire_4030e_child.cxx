/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include "acquire_4030e_child.h"
#include "aqprintf.h"
#include "dips_panel.h"
#include "varian_4030e.h"

Acquire_4030e_child::Acquire_4030e_child (int argc, char* argv[])
    : QCoreApplication (argc, argv)
{
    if (argc < 4) {
        aqprintf ("Error with commandline\n");
        exit (-1);
    }

    bool ok;
    this->idx = QString(argv[2]).toInt(&ok,10);
    if (!ok) {
        aqprintf ("Error with commandline\n");
        exit (-1);
    }

    aqprintf ("Child %s got request to open panel: %s\n", argv[2], argv[3]);
    this->idx = QString(argv[2]).toInt();
    this->dp = new Dips_panel;
    this->open_receptor (argv[3]);
}

Acquire_4030e_child::~Acquire_4030e_child ()
{
    if (this->dp) {
        delete this->dp;
    }
    if (this->vp) {
        delete this->vp;
    }
}

void 
Acquire_4030e_child::open_receptor (const char* path)
{
    int result;
    this->vp = new Varian_4030e (this->idx);
    result = vp->open_link (path);
    result = vp->disable_missing_corrections (result);
    if (result != HCP_NO_ERR) {
	aqprintf ("vp.open_receptor_link returns error (%d): %s\n", result, 
	    Varian_4030e::error_string(result));
        return;
        //return -1;
    }
    result = vp->check_link ();
    if (result != HCP_NO_ERR) {
	aqprintf ("vp.check_link returns error %d\n", result);
        vip_close_link();
        return;
        //return -1;
    }
    vp->print_sys_info ();

    result = vip_select_mode (vp->current_mode);
    vp->print_mode_info ();

    if (result != HCP_NO_ERR) {
        aqprintf ("vip_select_mode(%d) returns error %d\n", 
            vp->current_mode, result);
        vp->close_link ();
        //return -1;
    }
}

void 
Acquire_4030e_child::run()
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
