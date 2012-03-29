/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "HcpErrors.h"
#include "HcpFuncDefs.h"
#include "iostatus.h"
#include <QApplication>
#include <QThread>

#include "acquire_thread.h"
#include "advantech.h"
#include "dips_panel.h"
#include "varian_4030e.h"


// The string such as "A422-07" is the imager serial number
char *default_path_1 = "C:\\IMAGERs\\A422-07"; // Path to IMAGER tables
char *default_path_2 = "C:\\IMAGERs\\A663-11"; // Path to IMAGER tables

//----------------------------------------------------------------------
//  main
//----------------------------------------------------------------------
int 
main (int argc, char* argv[])
{
    Advantech advantech;
    char *paths[2];
    int choice = 0;

    QApplication app (argc, argv);

    printf ("Welcome to acquire_4030e\n");

    // Check for receptor path on the command line
    paths[0] = default_path_1;
    paths[1] = default_path_2;
    if (argc > 1) {
	paths[0] = argv[1];
    }
    if (argc > 2) {
	paths[1] = argv[2];
    }

    /* Start acquisition threads */
    QThread thread[2];
    Acquire_thread aq[2];
    for (int i = 0; i < 1; i++) {
        aq[i].idx = i;
        aq[i].open_receptor (paths[i]);
        aq[i].moveToThread (&thread[i]);
        thread[i].start ();
        QMetaObject::invokeMethod (&aq[i], "run", Qt::QueuedConnection);
    }

    /* Wait (forever) for threads to complete */
    app.exec();
    thread[0].wait();
    //thread[1].wait();

#if defined (commentout)
    Dips_panel dp;
    dp.open_panel (0, HIRES_IMAGE_HEIGHT, HIRES_IMAGE_WIDTH);

    /* Initialize link to panel */
    Varian_4030e vp;
    result = vp.open_receptor_link (path_1);
    result = vp.disable_missing_corrections (result);
    if (result != HCP_NO_ERR) {
	printf ("vp.open_receptor_link returns error (%d): %s\n", result, 
	    Varian_4030e::error_string(result));
        return -1;
    }

    result = vp.check_link ();
    if (result != HCP_NO_ERR) {
	printf ("vp.check_link returns error %d\n", result);
        vip_close_link();
        return -1;
    }

    vp.print_sys_info ();

    result = vip_select_mode (vp.current_mode);

    if (result != HCP_NO_ERR) {
        printf ("vip_select_mode(%d) returns error %d\n", 
            vp.current_mode, result);
        vip_close_link();
        return -1;
    }

    while (true) {
#if defined (commentout)
        /* Wait for generator expose request */
        while (!advantech.ready_for_expose ()) {
            Sleep (10);
        }
#endif
        /* Get frame from panel */
        printf ("Waiting for image.\n");
        vp.rad_acquisition (&dp);
    }

    vip_close_link();
#endif

    return 0;
}
