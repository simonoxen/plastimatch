/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <QApplication>
#include <QMessageBox>
#include <QProcess>
#include <QSystemTrayIcon>

#include "acquire_4030e_parent.h"
#include "acquire_4030e_window.h"
#include "kill.h"

Acquire_4030e_parent::Acquire_4030e_parent (int argc, char* argv[]) 
  : QApplication (argc, argv)
{
    printf ("Welcome to acquire_4030e\n");
    printf ("Creating child processes.\n");
    this->initialize (argc, argv);
}

Acquire_4030e_parent::~Acquire_4030e_parent ()
{
    /* Kill child processes */
    kill_process ("acquire_4030e.exe");

    /* Destroy window */
    delete this->window;
}

void 
Acquire_4030e_parent::initialize (int argc, char* argv[])
{
    char *paths[2];

    /* Check for system tray to store the UI */
    if (QSystemTrayIcon::isSystemTrayAvailable()) {
        printf ("System tray found.\n");
    }
    else {
        printf ("System tray not found.\n");
    }

    /* Start up main window */
    //this->setQuitOnLastWindowClosed (false);
    this->window = new Acquire_4030e_window;
    this->window->show ();

    /* Check for receptor path on the command line */
    if (argc > 1) {
        this->num_process = 1;
	paths[0] = argv[1];
    }
    if (argc > 2) {
	this->num_process = 2;
	paths[1] = argv[2];
    }

#if defined (commentout)
    /* Start child processes */
    for (int i = 0; i < this->num_process; i++) {
        QString program = argv[0];
        QStringList arguments;
	arguments << "--child" << QString("%1").arg(i).toUtf8() << paths[i];
	connect (&this->process[i], SIGNAL(readyReadStandardOutput()),
            this, SLOT(log_output()));
        this->process[i].start(program, arguments);
    }
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
