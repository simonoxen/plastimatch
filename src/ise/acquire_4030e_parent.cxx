/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <QApplication>
#include <QDebug>
#include <QMessageBox>
#include <QProcess>
#include <QSystemTrayIcon>
#include <QTimer>

#include "acquire_4030e_parent.h"
#include "acquire_4030e_window.h"
#include "advantech.h"
#include "kill.h"

Acquire_4030e_parent::Acquire_4030e_parent (int argc, char* argv[]) 
  : QApplication (argc, argv)
{
    printf ("Welcome to acquire_4030e\n");
    this->initialize (argc, argv);
}

Acquire_4030e_parent::~Acquire_4030e_parent ()
{
    /* Timer is deleted automatically */

    /* Detatch from advantech */
    this->advantech->relay_open (0);
    this->advantech->relay_open (3);
    this->advantech->relay_open (4);
    delete this->advantech;

    /* Destroy window */
    delete this->window;
}

void 
Acquire_4030e_parent::initialize (int argc, char* argv[])
{
    char *paths[2];

    /* Set up event handler for cleanup */
    connect (this, SIGNAL(aboutToQuit()), this, SLOT(about_to_quit()));

    /* Kill any leftover rogue processes */
    kill_rogue_processes ();

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

    /* Look for advantech device, spawn advantech thread */
    this->advantech = new Advantech;
    this->generator_prep = false;
    this->panel_select = false;
    this->advantech->relay_open (0);
    this->advantech->relay_open (3);
    this->advantech->relay_open (4);

    this->panel_timer = 0;

    /* Check for receptor path on the command line */
    if (argc > 1) {
        this->num_process = 1;
	paths[0] = argv[1];
    }
    if (argc > 2) {
	this->num_process = 2;
	paths[1] = argv[2];
    }

    /* Start child processes */
    printf ("Creating child processes.\n");
    for (int i = 0; i < this->num_process; i++) {
        QString program = argv[0];
        QStringList arguments;
	arguments << "--child" << QString("%1").arg(i).toUtf8() << paths[i];
	connect (&this->process[i], SIGNAL(readyReadStandardOutput()),
            this, SLOT(log_output()));
        this->process[i].start(program, arguments);
    }

    /* Spawn the timer for advantech */
    this->timer = new QTimer(this);
    connect (timer, SIGNAL(timeout()), this, SLOT(timer_event()));
    timer->start (50);
}

void 
Acquire_4030e_parent::kill_rogue_processes ()
{
    /* Kill child processes (either ours, or from previous instances) */
    kill_process ("acquire_4030e.exe");
}

void 
Acquire_4030e_parent::about_to_quit ()
{
    /* Kill children before we die */
    kill_rogue_processes ();
}

void 
Acquire_4030e_parent::log_output ()
{
    for (int i = 0; i < this->num_process; i++) {
        QByteArray result = process[i].readAllStandardOutput();
        QStringList lines = QString(result).split("\n");
        foreach (QString line, lines) {
            line = line.trimmed();
            if (line.isEmpty()) {
                continue;
            }
            line = QString("[%1] %2").arg(i).arg(line);

            /* Dump to window log */
            window->log_output (line);
            
            /* Dump to stdout */
            QByteArray line_ba = line.toAscii ();
            printf ("%s\n", (const char*) line_ba);
        }
    }
}

/* On STAR, panel 0 is axial, and panel 1 is g90 */
void 
Acquire_4030e_parent::timer_event ()
{
    /* On STAR, there is no distinction between prep & expose, i.e. there 
       is only prep signal. */
    bool gen_panel_select = this->advantech->read_bit (0);
    bool gen_prep_request = this->advantech->read_bit (1);
    bool gen_expose_request = this->advantech->read_bit (2);
    bool panel_0_ready = this->advantech->read_bit (3);
    bool panel_1_ready = this->advantech->read_bit (4);

    /* Write a debug message */
    if (gen_prep_request != this->generator_prep) {
        this->window->log_output (
            QString("[p] Generator status: %1 %2 %3 %4 %5")
            .arg(gen_panel_select).arg(gen_prep_request)
            .arg(gen_expose_request).arg(panel_0_ready).arg(panel_1_ready));
    }

    /* Check for new prep/expose request from generator */
    if (gen_prep_request && !this->generator_prep) {
	/* Set up timer */
	this->panel_timer = 0;

	/* Save state about which generator is active */
        this->generator_prep = true;
        this->panel_select = gen_panel_select;
    }

    if (gen_prep_request && this->generator_prep) {
	/* Check for timer wait complete */
	if (this->panel_timer <= 0) {
	    /* Close relay, asking panel to begin integration */
	    if (gen_panel_select == 0) {
		/* Axial */
		this->window->log_output (
		    QString("[p] Closing relay to panel: axial"));
		this->advantech->relay_close (3);
	    } else {
		/* G90 */
		this->window->log_output (
		    QString("[p] Closing relay to panel: g90"));
		this->advantech->relay_close (4);
	    }
	} else {
	    this->window->log_output (
		QString("[p] Timer value = %1").arg(this->panel_timer));
	    this->panel_timer --;
	}
    }

    /* Check if panel is ready */
    if (this->generator_prep) {
        /* Close relay on generator */
        if (this->panel_select == false && panel_0_ready) {
		this->window->log_output (
		    QString("[p] Closing relay to generator"));
		this->advantech->relay_close (0);
        }
        else if (this->panel_select == true && panel_1_ready) {
		this->window->log_output (
		    QString("[p] Closing relay to generator"));
		this->advantech->relay_close (0);
        }
        else if (panel_0_ready || panel_1_ready) {
            this->window->log_output (
                QString("[p] Warning, panel %1 was unexpectedly ready")
                .arg(panel_0_ready ? 0 : 1));
        }
	else {
	    this->window->log_output (
		    QString("[p] Neither panel is ready"));
	}
    }

    /* Check if generator prep request complete */
    if (this->generator_prep && !gen_prep_request) {
        this->advantech->relay_open (0);
        this->advantech->relay_open (3);
        this->advantech->relay_open (4);
        this->generator_prep = false;
	this->window->log_output (
		QString("[p] Reset this->generator_prep to false"));
    }
}
