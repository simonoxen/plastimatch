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

#include "acquire_4030e_window.h"

Acquire_4030e_window::Acquire_4030e_window ()
    : QMainWindow ()
{
    /* Sets up the GUI */
    setupUi (this);

    /* Set up the icon for the system tray */
    create_actions ();
    create_tray_icon ();
    set_icon ();
    tray_icon->show ();
}

void 
Acquire_4030e_window::set_icon ()
{
    tray_icon->setIcon (QIcon(":/red_ball.svg"));
    tray_icon->setToolTip (tr("Acquire 4030e"));
}

void 
Acquire_4030e_window::create_actions()
{
    minimize_action = new QAction(tr("Mi&nimize"), this);
    connect(minimize_action, SIGNAL(triggered()), this, SLOT(hide()));

    maximize_action = new QAction(tr("Ma&ximize"), this);
    connect(maximize_action, SIGNAL(triggered()), this, SLOT(showMaximized()));

    restore_action = new QAction(tr("&Restore"), this);
    connect(restore_action, SIGNAL(triggered()), this, SLOT(showNormal()));

    quit_action = new QAction(tr("&Quit"), this);
    connect(quit_action, SIGNAL(triggered()), qApp, SLOT(quit()));
}

void 
Acquire_4030e_window::create_tray_icon ()
{
    tray_icon_menu = new QMenu(this);
    tray_icon_menu->addAction (minimize_action);
    tray_icon_menu->addAction (maximize_action);
    tray_icon_menu->addAction (restore_action);
    tray_icon_menu->addSeparator ();
    tray_icon_menu->addAction (quit_action);

    tray_icon = new QSystemTrayIcon (this);
    tray_icon->setContextMenu (tray_icon_menu);
}
