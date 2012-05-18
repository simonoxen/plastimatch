/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <QApplication>
#include <QCloseEvent>
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

    /* Chuck some text into the text box for testing */
    log_viewer->appendPlainText ("Welcome to acquire_4030e.exe.");
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
    show_action = new QAction(tr("&Show"), this);
    connect(show_action, SIGNAL(triggered()), this, SLOT(showNormal()));

    quit_action = new QAction(tr("&Quit"), this);
    connect(quit_action, SIGNAL(triggered()), this, SLOT(request_quit()));
}

void 
Acquire_4030e_window::create_tray_icon ()
{
    tray_icon_menu = new QMenu(this);
    tray_icon_menu->addAction (show_action);
    tray_icon_menu->addSeparator ();
    tray_icon_menu->addAction (quit_action);

    tray_icon = new QSystemTrayIcon (this);
    tray_icon->setContextMenu (tray_icon_menu);

    connect (tray_icon, SIGNAL(activated(QSystemTrayIcon::ActivationReason)),
        this, SLOT(systray_activated(QSystemTrayIcon::ActivationReason)));
}

void 
Acquire_4030e_window::log_output (const QString& log)
{
    log_viewer->appendPlainText (log);
}

void 
Acquire_4030e_window::set_label_style (int panel_no, Label_style style)
{
    QString style_sheet;
    switch (style) {
	case LABEL_NOT_READY:
	    style_sheet = "QLabel { background-color : red; color : black; }";
	    break;
	case LABEL_ACQUIRING:
	    style_sheet = "QLabel { background-color : yellow; color : black; }";
	    break;
	case LABEL_READY:
	    style_sheet = "QLabel { background-color : green; color : black; }";
	    break;
    }
    if (panel_no == 0) {
	panel_1_status->setStyleSheet(style_sheet);
    }
    else {
	panel_2_status->setStyleSheet(style_sheet);
    }
}


void 
Acquire_4030e_window::set_label (int panel_no, const QString& log)
{
    if (panel_no == 0) {
	panel_1_status->setText(log);
    }
    else {
	panel_2_status->setText(log);
    }
}

void 
Acquire_4030e_window::request_quit ()
{
    tray_icon->hide ();
    qApp->quit();
}

void 
Acquire_4030e_window::systray_activated (
    QSystemTrayIcon::ActivationReason reason)
{
    switch (reason) {
    case QSystemTrayIcon::Trigger:
    case QSystemTrayIcon::DoubleClick:
    case QSystemTrayIcon::MiddleClick:
        this->show ();
        break;
    default:
        ;
    }
}

void 
Acquire_4030e_window::closeEvent(QCloseEvent *event)
{
    if (tray_icon->isVisible()) {
        hide();
        event->ignore();
    }
}
