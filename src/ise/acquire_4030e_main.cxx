/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <windows.h>
//#include "HcpErrors.h"
//#include "HcpFuncDefs.h"
//#include "iostatus.h"
#include <QApplication>
#include <QProcess>
#include <QThread>
//#include <QTest>

//#include "acquire_thread.h"
#include "advantech.h"
//#include "dips_panel.h"
//#include "varian_4030e.h"
#include "acquire_4030e_child.h"
#include "acquire_4030e_parent.h"
#include "kill.h"

class SleeperThread : public QThread {
public:
    static void msleep(unsigned long msecs) {
        QThread::msleep(msecs);
    }
};

//----------------------------------------------------------------------
//  main
//----------------------------------------------------------------------
int 
main (int argc, char* argv[])
{
    // The string such as "A422-07" is the imager serial number
    char *default_path_1 = "C:\\IMAGERs\\A422-07"; // Path to IMAGER tables
    char *default_path_2 = "C:\\IMAGERs\\A663-11"; // Path to IMAGER tables

    /* During debugging, use hard-coded path */
    if (argc < 2) {
        char** argv_tmp = (char**) malloc (3 * sizeof(char*));
        argv_tmp[0] = argv[0];
        argv_tmp[1] = default_path_1;
        argv_tmp[2] = default_path_2;
	argv = argv_tmp;
        //argc = 2;
        argc = 3;
    }

    if (argc > 1) {
        if (!strcmp (argv[1], "--child")) {
            /*** Child process ***/
            printf ("A child is born.\n");
            Acquire_4030e_child *child
                = new Acquire_4030e_child (argc, argv);

            /* Wait forever */
            printf ("Waiting forever.\n");
            child->run();
            printf ("Wait complete.\n");

#if defined (commentout)
            for (int i = 0; i < 10; i++) {
                printf ("A child is born.\n");
		fflush (stdout);
                SleeperThread::msleep(1000);
            }
#endif
        } else {
            /*** Parent process ***/

            /* Spawn child processes */
            Acquire_4030e_parent *parent 
                = new Acquire_4030e_parent (argc, argv);

            /* Wait forever */
            printf ("Waiting forever.\n");
            parent->exec ();
            printf ("Wait complete.\n");
        }
    } else {
        printf ("Usage: acquire_4030e image-path-1 [image-path-2]\n");
    }

#if defined (commentout)

    /* Start acquisition processes */
    QProcess process[2];
    for (int i = 0; i < 1; i++) {
        QString program = argv[0];
        QStringList arguments;
        arguments << "--child" << paths[i];
        process[i].start(program, arguments);

	connect (&process[i], SIGNAL(readyReadStandardOutput()),
            app, SLOT(log_output()));  

    }

    /* Wait (forever) for processes to complete */
    app.exec();
#endif

#if defined (commentout)
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
#endif

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
