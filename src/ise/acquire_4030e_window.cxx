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
#include <QMessageBox>
#include <QTimer>
#include <windows.h>


#include "acquire_4030e_window.h"
#include "acquire_4030e_parent.h"

Acquire_4030e_window::Acquire_4030e_window ()
    : QMainWindow ()
{
    /* Sets up the GUI */
    setupUi (this);

    /* Set up the icon for the system tray */
    create_actions ();
    create_tray_icon ();


	m_TimerReadyToQuit = new QTimer(this);	
	connect (m_TimerReadyToQuit, SIGNAL(timeout()), this, SLOT(TimerReadyToQuit_event()));
	//timer->start (50);  

    //set_icon ();
    //tray_icon->show ();

    /* Chuck some text into the text box for testing */
    log_viewer->appendPlainText ("Welcome to acquire_4030e.exe.");
}
void 
Acquire_4030e_window::set_icon (int idx, Label_style style)  // set_icon + tray icon
{   
    if (idx == 0)
    {
	switch (style) {
	case LABEL_NOT_READY:	    
	    tray_icon1->setIcon (QIcon(":/red_ball.png"));		
	    tray_icon1->setToolTip (tr("Acquire 4030e_Panel_1"));	    
	    break;
	case LABEL_ACQUIRING:	    
	    //tray_icon1->setIcon (QIcon(":/red_ball.svg"));
		tray_icon1->setIcon (QIcon(":/yellow_ball.png"));
	    tray_icon1->setToolTip (tr("Acquire 4030e_Panel_1"));	    
	    break;

	case LABEL_PREPARING:	    
		//tray_icon1->setIcon (QIcon(":/red_ball.svg"));
	    tray_icon1->setIcon (QIcon(":/orange_ball.png"));
	    tray_icon1->setToolTip (tr("Acquire 4030e_Panel_1"));	    
	    break;
	case LABEL_READY:	    
		//tray_icon1->setIcon (QIcon(":/green_ball.svg"));
	    tray_icon1->setIcon (QIcon(":/green_ball.png"));
	    tray_icon1->setToolTip (tr("AAcquire 4030e_Panel_1"));
	    break;
	}
	tray_icon1->show ();
    }    
    else if (idx ==1)
    {
	switch (style) {
	case LABEL_NOT_READY:	    
		tray_icon2->setIcon (QIcon(":/red_ball.png"));	    	    
		//tray_icon1->setIcon (QIcon(":/test_plain.svg"));	    	    
		tray_icon2->setToolTip (tr("Acquire 4030e_Panel_0"));	
	    break;
	case LABEL_ACQUIRING:	    
	    //tray_icon2->setIcon (QIcon(":/red_ball.svg"));
		tray_icon2->setIcon (QIcon(":/yellow_ball.png"));
	    tray_icon2->setToolTip (tr("Acquire 4030e_Panel_0"));	    
	    break;

	case LABEL_PREPARING:	    
	    tray_icon2->setIcon (QIcon(":/orange_ball.png"));
	    tray_icon2->setToolTip (tr("Acquire 4030e_Panel_0"));	    
	    break;
	case LABEL_READY:	    
	    tray_icon2->setIcon (QIcon(":/green_ball.png"));
	    tray_icon2->setToolTip (tr("AAcquire 4030e_Panel_0"));
	    break;
	}
	tray_icon2->show ();

    }
    //QString style_sheet;    
    //tray_icon2->show ();
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

    tray_icon2 = new QSystemTrayIcon (this); //YK
    tray_icon1 = new QSystemTrayIcon (this);    

    tray_icon2->setContextMenu (tray_icon_menu);
    tray_icon1->setContextMenu (tray_icon_menu);    

    connect (tray_icon2, SIGNAL(activated(QSystemTrayIcon::ActivationReason)),
        this, SLOT(systray_activated(QSystemTrayIcon::ActivationReason)));
    connect (tray_icon1, SIGNAL(activated(QSystemTrayIcon::ActivationReason)),
        this, SLOT(systray_activated(QSystemTrayIcon::ActivationReason)));
    
}

void 
Acquire_4030e_window::log_output (const QString& log)
{
    log_viewer->appendPlainText (log);
}

void 
Acquire_4030e_window::set_label_style (int panel_no, Label_style style) //change Label_style color
{
    QString style_sheet;
    switch (style) {
	case LABEL_NOT_READY:
	    style_sheet = "QLabel { background-color : rgba(255,0,0,200); color : white; font-weight: bold; font-size: 15px}";
	    break;
	case LABEL_ACQUIRING:
	    style_sheet = "QLabel { background-color : rgba(230,230,0,200); color : black; font-weight: bold;font-size: 15px}";
	    break;
	case LABEL_PREPARING:
	    style_sheet = "QLabel { background-color : rgba(255,153,0,200); color : black; font-weight: bold;font-size: 15px}";
	    break;
	case LABEL_READY:
	    style_sheet = "QLabel { background-color : rgba(0,255,0,200); color : black; font-weight: bold;font-size: 15px}";
	    break;
    }
    if (panel_no == 0) {
	panel_1_status->setStyleSheet(style_sheet); //panel_1_status: QLabel - HTML type input (str)
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

void Acquire_4030e_window::request_quit ()
{
	m_bSeqKillReady = true;
	((Acquire_4030e_parent*)qApp)->StartCommandTimer(0, Acquire_4030e_parent::KILL);	

	m_TimerReadyToQuit->start(1000);
	
	QTimer::singleShot(15000,this, SLOT(FinalQuit()));	
}

void Acquire_4030e_window::FinalQuit ()
{	
	//printf("Final Quit slot called\n");
	tray_icon1->hide ();
	tray_icon2->hide ();

	//this->close();
	qApp->closeAllWindows(); //qApp = Parent 
	qApp->quit(); //qApp = Parent	
}

void Acquire_4030e_window::systray_activated (
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

void Acquire_4030e_window::TimerReadyToQuit_event()
{
	if( ( (Acquire_4030e_parent*)qApp )->m_bChildReadyToQuit[0] && m_bSeqKillReady)
	{
		((Acquire_4030e_parent*)qApp)->StartCommandTimer(1, Acquire_4030e_parent::KILL);
		m_bSeqKillReady = false; //run only once
		return;
	}
	if (((Acquire_4030e_parent*)qApp)->m_bChildReadyToQuit[0] &&((Acquire_4030e_parent*)qApp)->m_bChildReadyToQuit[1]) //check all child process are closed and ready to quit
	{
		printf("Now program quits\n");
		m_TimerReadyToQuit->stop();
		FinalQuit ();
	}
	return;
}

void Acquire_4030e_window::UpdateLabel(int iPanelIdx, Label_style enStyle) // 0 based panel ID //called from child proc except the first time
{
    switch (enStyle)
    {
    case LABEL_NOT_READY:
	this->set_label_style (iPanelIdx, Acquire_4030e_window::LABEL_NOT_READY);
	this->set_label (iPanelIdx, "  Initializing");
	break;	

    case LABEL_ACQUIRING:
	this->set_label_style (iPanelIdx, Acquire_4030e_window::LABEL_ACQUIRING);
	this->set_label (iPanelIdx, "  Acquiring");
	break;

    case LABEL_PREPARING:
	this->set_label_style (iPanelIdx, Acquire_4030e_window::LABEL_PREPARING);
	this->set_label (iPanelIdx, "  Resetting");
	break;

    case LABEL_READY:
	this->set_label_style (iPanelIdx, Acquire_4030e_window::LABEL_READY);
	this->set_label (iPanelIdx, "  Ready");
	break;
    default:
	break;
    }
}

void 
Acquire_4030e_window::closeEvent(QCloseEvent *event)
{
    if (tray_icon1->isVisible()) {
        hide();        
    }
    if (tray_icon2->isVisible()) {
        hide();
        //event->ignore();
    }
    event->ignore();
}

void 
Acquire_4030e_window::RestartPanel_0 ()
{
       /* QMessageBox msgBox;	
	int test2 = ((Acquire_4030e_parent*)qApp)->test;
	QString strTitle = QString("%1").arg(test2);
	msgBox.setText(strTitle);
	msgBox.exec();*/
    //(Acquire_4030e_parent*)qApp->RestartChildProcess(0); //casting not work
    ((Acquire_4030e_parent*)qApp)->RestartChildProcess(0);
}

void 
Acquire_4030e_window::RestartPanel_1 ()
{    
    ((Acquire_4030e_parent*)qApp)->RestartChildProcess(1);
}

void Acquire_4030e_window::ShowPanelControl_0 ()
{    
    //QMessageBox msgBox;
    //QString strTitle = QString("test");
    //msgBox.setText(strTitle);    

    //((Acquire_4030e_parent*)qApp)->test = 5;
    //if (((Acquire_4030e_parent*)qApp)->m_dlgControl_0 != NULL) //This filter is not working
    //((Acquire_4030e_parent*)qApp)->ShowPanelControlWindow(0);    

    //((Acquire_4030e_parent*)qApp)->m_dlgControl_0->show();

	((Acquire_4030e_parent*)qApp)->StartCommandTimer(0, Acquire_4030e_parent::SHOWDLG);

	//m_dlgControl_0->show();
    //else
	//msgBox.exec();

}

void Acquire_4030e_window::ShowPanelControl_1 ()
{    
  //if (((Acquire_4030e_parent*)qApp)->m_dlgControl_1 != NULL) //doens't work. maybe cannot access to the address
    //((Acquire_4030e_parent*)qApp)->m_dlgControl_1->show();
    //((Acquire_4030e_parent*)qApp)->m_dlgControl_1->show();
	((Acquire_4030e_parent*)qApp)->StartCommandTimer(1, Acquire_4030e_parent::SHOWDLG);
}

//void Acquire_4030e_window::ForcedQuit()
//{
//	tray_icon1->hide ();
//	tray_icon2->hide ();
//	qApp->quit();
//}