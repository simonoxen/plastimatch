/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <QApplication>
#include <QProcess>

#include "acquire_4030e_parent.h"

Acquire_4030e_parent::Acquire_4030e_parent (int argc, char* argv[]) 
  : QApplication (argc, argv)
{
    printf ("Welcome to acquire_4030e\n");
    printf ("Creating child processes.\n");
    this->initialize (argc, argv);
}

void 
Acquire_4030e_parent::initialize (int argc, char* argv[])
{
    char *paths[2];

    // Check for receptor path on the command line
    if (argc > 1) {
        this->num_process = 1;
	paths[0] = argv[1];
    }
    if (argc > 2) {
	this->num_process = 2;
	paths[1] = argv[2];
    }

    /* Start acquisition processes */
    for (int i = 0; i < this->num_process; i++) {
        QString program = argv[0];
        QStringList arguments;
	arguments << "--child" << QString("%1").arg(i).toUtf8() << paths[i];
	connect (&this->process[i], SIGNAL(readyReadStandardOutput()),
            this, SLOT(log_output()));
        this->process[i].start(program, arguments);
    }

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
}

void 
Acquire_4030e_parent::log_output ()
{
    for (int i = 0; i < this->num_process; i++) {
        QByteArray result = process[i].readAllStandardOutput();
        QStringList lines = QString(result).split("\n");
        foreach (QString line, lines) {
            line = line.trimmed();
            if (!line.isEmpty()) {
                QByteArray line_ba = line.toAscii ();
                printf ("[%d] %s\n", i, (const char*) line_ba);
            }
        }
    }
}
