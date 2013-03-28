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
#include <QTime>
#include <windows.h>
#include <QFileDialog>

#include "acquire_4030e_parent.h"
//#include "acquire_4030e_window.h"
#include "advantech.h"
#include "kill.h"

Acquire_4030e_parent::Acquire_4030e_parent (int argc, char* argv[]) 
: QApplication (argc, argv)
{
	printf ("Welcome to acquire_4030e\n");
	this->initialize (argc, argv);    

	//m_dlgControl_0 = NULL;
	// m_dlgControl_1 = NULL;
	m_bWaitingForChildResponse[0] = false;
	m_bWaitingForChildResponse[1] = false;

	m_enPrevSentCommand[0] = DUMMY;   	
	m_enPrevSentCommand[1] = DUMMY;   	

	m_bChildReadyToQuit[0] = true; //exit function can be called in initial status
	m_bChildReadyToQuit[1] = true;

}

Acquire_4030e_parent::~Acquire_4030e_parent () //not called!!
{
	/* Timer is deleted automatically */

	/* Detatch from advantech */
	this->advantech->relay_open (0);
	this->advantech->relay_open (3);
	this->advantech->relay_open (4);
	delete this->advantech;

	/* Destroy window */
	delete this->window;
	//delete m_dlgControl_0;
	//delete m_dlgControl_1;
}

void 
Acquire_4030e_parent::initialize (int argc, char* argv[])
{
	char *paths[2];

	/* Set up event handler for cleanup */
	connect (this, SIGNAL(aboutToQuit()), this, SLOT(about_to_quit())); //whenever quit signal --> run quit

	/* Kill any leftover rogue processes */
	kill_rogue_processes (); //delete same process

	/* Check for system tray to store the UI */
	if (QSystemTrayIcon::isSystemTrayAvailable()) {
		printf ("System tray found.\n");
	}	
	else {
		printf ("System tray not found.\n");
	}

	/* Start up main window */
	//this->setQuitOnLastWindowClosed (false);
	this->window = new Acquire_4030e_window();  //GUI-linked window
	//this->window2 = new Acquire_4030e_window();  //GUI-linked window

	this->window->setWindowTitle("Acquire 4030e v1.0 by MGH RO Physics Team");
	//this->window->setWindowTitle("acquire_4030e");

	this->window->UpdateLabel(0, Acquire_4030e_window::LABEL_NOT_READY);
	this->window->UpdateLabel(1, Acquire_4030e_window::LABEL_NOT_READY);
	this->window->set_icon(0,Acquire_4030e_window::LABEL_NOT_READY);// Tray should be shown after Label is updated.
	this->window->set_icon(1,Acquire_4030e_window::LABEL_NOT_READY);
	this->window->show ();
	//this->window2->show ();

	//m_dlgControl_0 = new Acquire_4030e_DlgControl();    
	//m_dlgControl_0->setWindowTitle("Panel Control Dialog: Panel 0");
	//m_dlgControl_0->m_iPanelIdx = 0;

	//m_dlgControl_1 = new Acquire_4030e_DlgControl();    
	//m_dlgControl_1->setWindowTitle("Panel Control Dialog: Panel 1");
	//m_dlgControl_1->m_iPanelIdx = 1;

	//this->m_dlgControl_0->show ();
	//this->m_dlgControl_1->show ();

	//QDialog *gamatosdialog = new QDialog; it works!
	//gamatosdialog->show();

	/*  QMessageBox msgBox;
	QString strTitle = QString("dlgControl has been made");
	msgBox.setText(strTitle);
	msgBox.exec();*/

	/* Look for advantech device, spawn advantech thread */
	this->advantech = new Advantech;
	this->generator_state = WAITING;
	this->panel_select = false;
	this->advantech->relay_open (0);
	this->advantech->relay_open (3);
	this->advantech->relay_open (4);

	this->panel_timer = 0;

	/* Check for receptor path on the command line */
	if (argc > 1) {
		this->num_process = 1;
		paths[0] = argv[1]; //C:\Imagers\...
	}
	if (argc > 2) {
		this->num_process = 2;
		paths[1] = argv[2];
	}

	/* Start child processes */
	printf ("Creating child processes.\n");
	for (int i = 0; i < this->num_process; i++) {
		m_program[i] = argv[0]; //exe file path
		m_arguments[i].clear();
		m_arguments[i] << "--child" << QString("%1").arg(i).toUtf8() << paths[i];
		connect (&this->process[i], SIGNAL(readyReadStandardOutput()),
			this, SLOT(poll_child_messages()));


		//if (i == 0) //YKTEMP: temp code only panel 0 go!
		this->process[i].start(m_program[i], m_arguments[i]);
	}

	m_enPanelStatus[0] = Acquire_4030e_window::LABEL_NOT_READY;
	m_enPanelStatus[1] = Acquire_4030e_window::LABEL_NOT_READY;

	m_bPleoraErrorHasBeenOccurredFlag[0] = false;
	m_bPleoraErrorHasBeenOccurredFlag[1] = false;

	/* Spawn the timer for polling devices */
	this->timer = new QTimer(this);
	connect (timer, SIGNAL(timeout()), this, SLOT(timer_event()));
	timer->start (50);  


	this->timerCommandToChild[0] = new QTimer(this);
	connect (timerCommandToChild[0], SIGNAL(timeout()), this, SLOT(timerCommandToChild0_event()));

	this->timerCommandToChild[1] = new QTimer(this);
	connect (timerCommandToChild[1], SIGNAL(timeout()), this, SLOT(timerCommandToChild1_event()));



	this->timerAboutToQuit= new QTimer(this);
	connect (timerAboutToQuit, SIGNAL(timeout()), this, SLOT(timerAboutToQuit_event()));

	//this->timerSysTray = new QTimer(this);
	//connect (timerSysTray, SIGNAL(timeout()), this, SLOT(timerSysTray_event()));
	//timerSysTray->start (5000); //every 5 s

	//init log file
	m_strLogFilePath = DEFAULT_LOGFILE_PATH;

	//YK_FUTURE
	//m_strLogFilePath = from option file

	m_strLogFilePath.append("acquire4030e_log_");

	QDate crntDate = QDate::currentDate ();
	QString dateFormat = "MM_dd_yyyy";
	QString strDate = crntDate.toString(dateFormat);

	QTime crntTime = QTime::currentTime ();
	QString timeFormat = "hh_mm_";
	QString strTime = crntTime.toString(timeFormat);

	m_strLogFilePath.append(strTime);
	m_strLogFilePath.append(strDate);

	m_strLogFilePath.append(".txt");

	m_logFout.open(m_strLogFilePath.toLocal8Bit().constData());

	if (m_logFout.fail())
	{
		printf("Cannot open log output file");
	}
}

void 
Acquire_4030e_parent::kill_rogue_processes ()
{
	/* Kill child processes (either ours, or from previous instances) */
	//window->log_output("YKTEMP: program is shutting down2");
	kill_process ("acquire_4030e.exe");
}

void 
Acquire_4030e_parent::about_to_quit () //called from window->FinalQuit() as well
{
	//*StartCommandTimer(0, Acquire_4030e_parent::KILLANDEXIT);


	//StartCommandTimer(1, Acquire_4030e_parent::KILLANDEXIT);

	//Call destructor of child
	//this->SendCommandToChild(0,CommandToChild::KILL);
	//this->SendCommandToChild(1,CommandToChild::KILL);
	//case GAIN_CORR_APPLY_OFF:
	// process[idx].write("PCOMMAND_GAINCORRAPPLYOFF\n");//write stdin of each process
	// break;

	//process[idx].write("PCOMMAND_GAINCORRAPPLYOFF\n");//write stdin of each process
	// break;

	//*this->timerAboutToQuit->start(2000);
	timer->stop();

	if (advantech != NULL)
	{
		this->advantech->relay_open (0);
		this->advantech->relay_open (3);
		this->advantech->relay_open (4);	

		/* Destroy window */
		//yk: WHEN KILL COMMAND OCCURRED			
		delete advantech;	
		advantech = NULL;
	}	

	if (window != NULL)
	{
		window->tray_icon1->hide ();
		window->tray_icon2->hide ();
		delete window;
		window = NULL;
	}

	//msgBox.setText("test4");
	//msgBox.exec();

	//after response of child arrives, quit the parent

	//disconnect

	disconnect (&this->process[0], SIGNAL(readyReadStandardOutput()), this, SLOT(poll_child_messages()));
	disconnect (&this->process[1], SIGNAL(readyReadStandardOutput()), this, SLOT(poll_child_messages()));

	/* Kill children before we die */
	kill_rogue_processes (); //not safe quit (for child)

	m_logFout.close();	
}

//void Acquire_4030e_parent::quit()
//{
//	if (true)
//		QApplication::closeAllWindows()::quit();
//
//	return;
//}


void 
Acquire_4030e_parent::log_output (const QString& log)
{
	/* Dump to window log */
	window->log_output (log);

	if (!m_logFout.fail())
		m_logFout << log.toLocal8Bit().constData() << std::endl;

	/* Dump to stdout */
	QByteArray ba = log.toAscii ();
	printf ("%s\n", (const char*) ba);
}

void 
Acquire_4030e_parent::poll_child_messages () //often called even window is closed
{	

	for (int i = 0; i < this->num_process; i++)
	{
		QByteArray result = process[i].readAllStandardOutput();
		QStringList lines = QString(result).split("\n");
		foreach (QString line, lines)
		{
			line = line.trimmed();
			if (line.isEmpty()) {
				continue;
			}
			if (window == NULL)
				break; //do nothing

			QTime time = QTime::currentTime();    
			QString strTime = time.toString("\t@hh:mm:ss.zzz");
			line = QString("[%1] %2").arg(i).arg(line);
			line.append(strTime);
			this->log_output (line);

			/********************* YKP 0126 2013 added ****************************/
			//cannot directly receive from acquire4030e_children --> different process
			//connect has been done btw. Qproc (not child) and parents 
			//It seems that the only way to upward-communicate of child is this status msg.
			//Using string-process, parents can action according to child's (Qprocess of child) feedback

			//Assumming that process number = panel number
			if (line.contains("PRESPP"))
			{
				m_bWaitingForChildResponse[i] = false; //start to send dummy message

				if (line.contains("OPENPANEL")) //PRESPP means aleady close the link
				{
					m_enPanelStatus[i] = Acquire_4030e_window::LABEL_NOT_READY;		
					m_bPleoraErrorHasBeenOccurredFlag[i] = false;		    
				}
				//else if (line.contains("KILLANDEXIT")) //PRESPP means aleady close the link
				//{
				//	//printf ("YK: Before kill %d.\n", i);
				//	//only 0 panel receive this signal
				//	if (i == 0)
				//	{
				//		process[0].close();						
				//		m_enPanelStatus[0] = Acquire_4030e_window::LABEL_NOT_READY;		
				//		m_bPleoraErrorHasBeenOccurredFlag[0] = false;

				//		process[1].close();
				//		m_enPanelStatus[1] = Acquire_4030e_window::LABEL_NOT_READY;						
				//		m_bPleoraErrorHasBeenOccurredFlag[1] = false;

				//		this->about_to_quit();					
				//	}
				//}
				else if (line.contains("KILL")) //PRESPP means aleady close the link
				{
					//printf ("YK: Before kill %d.\n", i);
					process[i].close(); //call destructor 
					m_enPanelStatus[i] = Acquire_4030e_window::LABEL_NOT_READY;		
					m_bPleoraErrorHasBeenOccurredFlag[i] = false;
					m_bChildReadyToQuit[i] = true;

				}
				else if (line.contains("RESTART"))
				{
					m_enPanelStatus[i] = Acquire_4030e_window::LABEL_NOT_READY;		
					m_bPleoraErrorHasBeenOccurredFlag[i] = false;
					RestartChildProcess(i);
				}
				/*else if (line.contains("STOPLOOP"))
				{
				m_enPanelStatus[i] = Acquire_4030e_window::LABEL_PREPARING;				 
				}
				else if (line.contains("RESUMELOOP"))
				{		    
				m_enPanelStatus[i] = Acquire_4030e_window::LABEL_PREPARING;				 
				}
				else if (line.contains("OPENPANEL"))
				{		    
				m_enPanelStatus[i] = Acquire_4030e_window::LABEL_PREPARING;				 
				}
				else if (line.contains("CLOSEPANEL"))
				{		    
				m_enPanelStatus[i] = Acquire_4030e_window::LABEL_NOT_READY;				 
				}*/
			}
			else if (line.contains("PSTAT0"))
			{
				m_enPanelStatus[i] = Acquire_4030e_window::LABEL_NOT_READY;
				//m_bPleoraErrorHasBeenOccurredFlag[i] = false;
			}
			else if (line.contains("PSTAT1"))
			{
				m_enPanelStatus[i] = Acquire_4030e_window::LABEL_PREPARING;
				m_bPleoraErrorHasBeenOccurredFlag[i] = false;
			}	    
			else if (line.contains("PSTAT3"))
			{
				this->m_enPanelStatus[i] = Acquire_4030e_window::LABEL_READY;		
				m_bPleoraErrorHasBeenOccurredFlag[i] = false;

				m_bChildReadyToQuit[0] = false; //cannot exit without quit the child process safely
				m_bChildReadyToQuit[1] = false;

			}	
			else if (line.contains("PSTAT4"))
			{
				this->m_enPanelStatus[i] = Acquire_4030e_window::LABEL_PREPARING;				
			}
			else if (line.contains("READY FOR X-RAYS") || line.contains("PSTAT3"))
			{
				this->m_enPanelStatus[i] = Acquire_4030e_window::LABEL_READY;		
				m_bPleoraErrorHasBeenOccurredFlag[i] = false;
			}
			else if (line.contains("Waiting for Complete_Frames")) //called from wait on num frames first
			{
				this->m_enPanelStatus[i] = Acquire_4030e_window::LABEL_PREPARING;
				m_bPleoraErrorHasBeenOccurredFlag[i] = false;
			}
			else if (line.contains("RESTART_PROCESS")) //
			{		
				m_enPanelStatus[i] = Acquire_4030e_window::LABEL_NOT_READY;		
				m_bPleoraErrorHasBeenOccurredFlag[i] = false;
				RestartChildProcess(i);
			}	    
			else if(line.contains("open_receptor_link returns error")) //error during start
			{
				this->m_enPanelStatus[i] = Acquire_4030e_window::LABEL_NOT_READY; //go to init status->red
				//m_bPanelReady[i] = false;

				//messageBox
				QMessageBox msgBox;
				//QString str = "Child cannot be created";
				QString strTitle = QString("Panel %1 Error").arg(i);
				msgBox.setWindowTitle(strTitle);

				QString str = QString("Panel #%1: Cable connection error during the start. Ethernet cable or power cable may have been unplugged.\n [1] Check cable connection\n [2] Re-plug in\n [3] Wait for at least 10 s then push [Retry] button.\n # if you click [Close] then exit from the program.\n").arg(i);
				msgBox.setText(str);
				msgBox.setStandardButtons(QMessageBox::Retry|QMessageBox::Close);
				msgBox.setDefaultButton(QMessageBox::Retry);
				int result = msgBox.exec();
				if (result == QMessageBox::Retry)
				{
					RestartChildProcess(i);
				}
				else if (result == QMessageBox::Close)
				{
					this->about_to_quit();
					exit(0);
				}
			}
			else if(line.contains("Pleora Error")) //error during running 1)ethernet, 2) power
			{
				this->m_enPanelStatus[i] = Acquire_4030e_window::LABEL_NOT_READY; //go to init status->red
				m_bPleoraErrorHasBeenOccurredFlag[i] = true;
				//messageBox
				QMessageBox msgBox;
				QString strTitle = QString("Panel %1 Error").arg(i);
				msgBox.setWindowTitle(strTitle);

				QString str = QString("Panel #%1: Cable connection error. Ethernet cable or power cable may have been unplugged.\n [1] Check cable connection\n [2] Re-plug in\n [3] Wait for at least 10 s then push [Retry] button.\n # if you click [Close] then exit from the program.\n").arg(i);		 

				msgBox.setText(str);
				msgBox.setStandardButtons(QMessageBox::Retry|QMessageBox::Close);
				msgBox.setDefaultButton(QMessageBox::Retry);
				int result = msgBox.exec();
				if (result == QMessageBox::Retry)
				{
					RestartChildProcess(i);
				}
				else if (result == QMessageBox::Close)
				{
					this->about_to_quit();
					exit(0);
				}		
			}
			else if(line.contains("State Error")) //Occassionally, when power was re-plugged in
			{
				if (m_bPleoraErrorHasBeenOccurredFlag[i])// only when PleoraError has been occurred just before --> restart is required
				{
					this->m_enPanelStatus[i] = Acquire_4030e_window::LABEL_NOT_READY; //go to init status->red

					QMessageBox msgBox;
					QString strTitle = QString("Panel %1 Error").arg(i);
					msgBox.setWindowTitle(strTitle);

					QString str = QString("Panel #%1: State error. This will occur when power cable replugged in.\n Push [retry] button to restart the child process\n # if you click [Close] then exit from the program.\n").arg(i);		 
					msgBox.setText(str);
					msgBox.setStandardButtons(QMessageBox::Retry|QMessageBox::Close);
					msgBox.setDefaultButton(QMessageBox::Retry);
					int result = msgBox.exec();
					if (result == QMessageBox::Retry)
					{
						RestartChildProcess(i);
					}
					else if (result == QMessageBox::Close)
					{
						//	this->log_output("YKTEMP: program is shutting down1");
						this->about_to_quit();
						exit(0);
					}
				}		
			}
			/***********************************************************/
		}
	}
}

bool Acquire_4030e_parent::RestartChildProcess(int idx)
{
	if (idx >=2)
		return false;

	//YK: it didn't work
	//if (!process[idx].atEnd()) //if program is running //Actually the process already has been terminated before this message.
	//{
	//process[idx].kill();
	//}
	//printf ("YK: Before kill %d.\n", idx);
	process[idx].close();
	//process[idx].terminate();//cannot terminate the process //send some event on QApp?
	//process[idx].aboutToClose();
	//printf ("YK: Re-creating child process %d.\n", idx);
	m_enPanelStatus[idx] = Acquire_4030e_window::LABEL_NOT_READY;
	connect (&this->process[idx], SIGNAL(readyReadStandardOutput()),
		this, SLOT(poll_child_messages()));

	this->process[idx].start(m_program[idx], m_arguments[idx]); //temporary deleted //Restart the main

	m_bChildReadyToQuit[idx] = false;
	return true;
}
/* On STAR, panel 0 is axial, and panel 1 is g90 */
void 
Acquire_4030e_parent::timer_event () //will be runned from the first time.
{
	//YK
	UpdateLableStatus(); //based-on m_bPanelReady status; //every 50 ms

	if (!m_bWaitingForChildResponse[0]) //if m_bWaitingForChildResponse == false send dummy char
	{
		process[0].write("\n");//write stdin of each process    		
	}
	if (!m_bWaitingForChildResponse[1]) //if m_bWaitingForChildResponse == false send dummy char
	{
		process[1].write("\n");//write stdin of each process    		
	}	

	//process[0].write(" ");//write stdin of each process    
	//process[1].write(" "); //no in reference but work!

	/* On STAR, there is no distinction between prep & expose, i.e. there 
	is only prep signal. */    

	int res0 = advantech->read_bit (0);
	int res1 = advantech->read_bit (1);
	int res2 = advantech->read_bit (2);
	int res3 = advantech->read_bit (3);
	int res4 = advantech->read_bit (4);

	if (res0 == Advantech::STATE_ERROR ||
		res1 == Advantech::STATE_ERROR ||
		res2 == Advantech::STATE_ERROR ||
		res3 == Advantech::STATE_ERROR ||
		res4 == Advantech::STATE_ERROR)
		return;

	bool gen_panel_select = (bool)res0;
	//= this->advantech->read_bit (0);  //0:false -> panel_0, 1:true ->panel_1
	bool gen_prep_request = (bool)res1;;
	//= this->advantech->read_bit (1);	/* Ignored on STAR */
	bool gen_expose_request = (bool)res2;;
	//= this->advantech->read_bit (2); //beam-on signal to advantech
	bool panel_0_ready = (bool)res3;;
	//= this->advantech->read_bit (3); //signal from panel
	bool panel_1_ready = (bool)res4;;
	//= this->advantech->read_bit (4); //signal from panel    


	/* Write a debug message */
	if (gen_expose_request) {
		if (this->generator_state == WAITING || panel_0_ready || panel_1_ready) {
			this->log_output (
				QString("[p] Generator status: %1 %2 %3 %4 %5")
				.arg(gen_panel_select).arg(gen_prep_request)
				.arg(gen_expose_request).arg(panel_0_ready).arg(panel_1_ready));	
		}
	}

	/* Check for new prep/expose request from generator */
	if (gen_expose_request && this->generator_state == WAITING) // if panel is not ready but only gen state and expose request should not trigger the relay.//at least do nothing with panel when it is not ready
	{
		/* Save state about which generator is active */
		this->panel_select = gen_panel_select;

		/* Close relay, asking panel to begin integration */
		if (gen_panel_select == 0)
		{
			/* Axial */
			this->log_output (
				QString("[p] Closing relay to panel: axial"));
			this->advantech->relay_close (3); //close = connected
		}	
		else if (gen_panel_select == 1)
		{
			/* G90 */
			this->log_output (
				QString("[p] Closing relay to panel: g90"));
			this->advantech->relay_close (4);
		}	
		this->generator_state = EXPOSE_REQUEST;
	}

	/* Check if panel is ready */
	if (gen_expose_request && this->generator_state == EXPOSE_REQUEST) {
		/* When panel is ready, close relay on generator */
		if (this->panel_select == false && panel_0_ready) {
			this->log_output (
				QString("[p] Closing relay to generator"));
			this->advantech->relay_close (0);
			this->generator_state = EXPOSING;
		}
		else if (this->panel_select == true && panel_1_ready) {
			this->log_output (
				QString("[p] Closing relay to generator"));
			this->advantech->relay_close (0);
			this->generator_state = EXPOSING;
		}
		else if (panel_0_ready || panel_1_ready) {
			this->log_output (
				QString("[p] Warning, panel %1 was unexpectedly ready")
				.arg(panel_0_ready ? 0 : 1));
		}
		else {
			this->log_output (
				QString("[p] Waiting for panel %1").arg(this->panel_select));	    	    
		}	
		//printf("panel_select: %d, panel_0_ready: %d, panel_1_ready: %d\n\n",panel_select, panel_0_ready, panel_1_ready);
	}    

	/* Check if generator prep request complete */
	if (!gen_expose_request) { //when exposure is over then relay_open: allow panel to read data
		this->advantech->relay_open (0); //PANEL SELECTION	
		this->advantech->relay_open (3); //I'm Done signal
		this->advantech->relay_open (4);//yk: SEND SOME TRIGGERING TO PANEL?

		if (this->generator_state != WAITING) {
			this->log_output (
				QString("[p] Reset generator state to WAITING."));
		}
		this->generator_state = WAITING;
	}

	if (generator_state == EXPOSING)
	{
		if (gen_panel_select == false) // panel_0
		{
			this->m_enPanelStatus[0] = Acquire_4030e_window::LABEL_ACQUIRING;
		}
		if (gen_panel_select == true) // panel_0
		{
			this->m_enPanelStatus[1] = Acquire_4030e_window::LABEL_ACQUIRING;
		}
	}
	//other lable style is coverned by child process
}


void 
Acquire_4030e_parent::UpdateLableStatus()
{
	for (int i = 0 ; i<MAX_PANEL_NUM_YK ; i++)
	{
		this->window->UpdateLabel(i,this->m_enPanelStatus[i]);
		this->window->set_icon((1-i),this->m_enPanelStatus[i]);
	} 
}
//
//void Acquire_4030e_parent::GetPanelInfo(int idx) //Msg box
//{
//    //QTime time = QTime::currentTime();    
//    //QString strTime = time.toString("@hh:mm:ss.zzz\n"); 
//
//    //QString strSendPanel_0 = "MsgFromMainProcessToPanel_0: " + strTime + "\n";
//    //QString strSendPanel_1= "MsgFromMainProcessToPanel_1: " + strTime + "\n";    
//
//    //strSendPanel_0 =" "; //some problem occurred!!! should be  at least length
//    //strSendPanel_1 =" ";
//
//    QString strRequestForInfo = "DISPLAY_PANEL_INFO\n";
//    process[idx].write(strRequestForInfo.toStdString().c_str());//write stdin of each process
//    //process[0].closeWriteChannel(); //this will close the channel 
//    //process[1].write(strSendPanel_1.toStdString().c_str());//no in reference but work!
//}

//Button click
void Acquire_4030e_parent::StartCommandTimer(int idx, Acquire_4030e_parent::CommandToChild enCommand)
{
	if (m_bWaitingForChildResponse[idx])
	{
		QString strLog = "Cannot send command to child because still waiting for response";
		log_output(strLog);
		return;
	}    
	SendCommandToChild(idx, enCommand); //Msg box  
	//Sleep(2000);
	Sleep(2000); //minimum delay

	//m_iPrevSentPanelIdx = idx;
	m_enPrevSentCommand[idx] = enCommand;
	m_iMaxResendTryCnt[idx] = 0;
	timerCommandToChild[idx]->start (2000);  
}

void Acquire_4030e_parent::timerCommandToChild0_event()
{
	if (!m_bWaitingForChildResponse[0]) //when child response has arrived
	{
		timerCommandToChild[0]->stop();
		return;
	}
	if (m_iMaxResendTryCnt[0] > MAX_RESEND_TRY)
	{
		timerCommandToChild[0]->stop();
		m_bWaitingForChildResponse[0] = false;
		return;
	}

	m_iMaxResendTryCnt[0]++;    
	SendCommandToChild(0, m_enPrevSentCommand[0]); //write at process
}
void Acquire_4030e_parent::timerCommandToChild1_event()
{
	if (!m_bWaitingForChildResponse[1]) //when child response has arrived
	{
		timerCommandToChild[1]->stop();
		return;
	}
	if (m_iMaxResendTryCnt[1] > MAX_RESEND_TRY)
	{
		timerCommandToChild[1]->stop();
		m_bWaitingForChildResponse[1] = false;
		return;
	}

	m_iMaxResendTryCnt[1]++;
	SendCommandToChild(1, m_enPrevSentCommand[1]); //write at process
}

void Acquire_4030e_parent::timerAboutToQuit_event()
{
	//if (!m_bWaitingForChildResponse 
	//	&& m_enPanelStatus[0] == Acquire_4030e_window::LABEL_NOT_READY
	//	&& m_enPanelStatus[1] == Acquire_4030e_window::LABEL_NOT_READY) //when child response has arrived
	//{
	//	timerAboutToQuit->stop ();  

	//	this->advantech->relay_open (0);
	//	this->advantech->relay_open (3);
	//	this->advantech->relay_open (4);

	//	/* Destroy window */
	//	//yk: WHEN KILL COMMAND OCCURRED	

	//	delete this->advantech;
	//	delete this->window;

	//	m_logFout.close();
	//	//after response of child arrives, quit the parent
	//	/* Kill children before we die */
	//	kill_rogue_processes ();				
	//}	
	return;
}


//this func will not skipped at any time.
void Acquire_4030e_parent::SendCommandToChild(int idx, Acquire_4030e_parent::CommandToChild enCommand) //Msg box
{    
	// if process is not ready, nothing.
	//RESTART only when the process is killed
	//int result1 = process[0].isOpen();
	//int result2 = process[1].isOpen();
	if (!process[idx].isOpen() && enCommand == RESTART) //if even the process has not been started,
	{
		RestartChildProcess(idx);
		return;
	}
	else if (!process[idx].isOpen() && enCommand != RESTART)
	{
		return;
	}    
	m_enPrevSentCommand[idx] = enCommand;

	m_bWaitingForChildResponse[idx] = true;       

	switch (enCommand)
	{
		//case OPEN_PANEL:
		//	process[idx].write("PCOMMAND_OPENPANEL\n");//write stdin of each process
		//	break;
		//case CLOSE_PANEL:
		//	process[idx].write("PCOMMAND_CLOSEPANEL\n");//write stdin of each process
		//	break;
	case KILL:
		process[idx].write("PCOMMAND_KILL\n");//write stdin of each process
		//case KILLANDEXIT:
		//	process[idx].write("PCOMMAND_KILLANDEXIT\n");//write stdin of each process
	case RESTART:
		process[idx].write("PCOMMAND_RESTART\n");//write stdin of each process
		break;	
		//case GET_PANEL_INFO:
		//	process[idx].write("PCOMMAND_GETINFO\n");//write stdin of each process
		//	break;
		//case STOP_LOOP:
		//	//log_output (QString("Stop loop clicked"));
		//	process[idx].write("PCOMMAND_STOPLOOP\n");//write stdin of each process
		//	break;
		//case RESUME_LOOP:
		//	//log_output (QString("Resume loop clicked"));
		//	process[idx].write("PCOMMAND_RESUMELOOP\n");//write stdin of each process
		//	break;
		//case SOFTWARE_HANDSHAKING_ENABLE:
		//	process[idx].write("PCOMMAND_SOFTHANDSHAKING\n");//write stdin of each process
		//	break;

		//case SOFTWARE_BEAM_ON:
		//	process[idx].write("PCOMMAND_SOFTBEAMON\n");//write stdin of each process
		//	break;
		//case HARDWARE_HANDSHAKING_ENABLE:
		//	process[idx].write("PCOMMAND_HARDHANDSHAKING\n");//write stdin of each process
		//	break;	
		//case GET_DARK_FIELD_IMAGE:
		//	process[idx].write("PCOMMAND_GETDARK\n");//write stdin of each process
		//	break;
		//	/*DARKCORRAPPLYON,
		//	DARK_CORR_APPLY_OFF,

		//	GAIN_CORR_APPLY_ON,
		//	GAIN_CORR_APPLY_OFF,*/
		//case DARK_CORR_APPLY_ON:
		//	process[idx].write("PCOMMAND_DARKCORRAPPLYON\n");//write stdin of each process
		//	break;
		//case DARK_CORR_APPLY_OFF:
		//	process[idx].write("PCOMMAND_DARKCORRAPPLYOFF\n");//write stdin of each process
		//	break;
		//case GAIN_CORR_APPLY_ON:
		//	process[idx].write("PCOMMAND_GAINCORRAPPLYON\n");//write stdin of each process
		//	break;
		//case GAIN_CORR_APPLY_OFF:
		//	process[idx].write("PCOMMAND_GAINCORRAPPLYOFF\n");//write stdin of each process
		//	break;
	case SHOWDLG:
		//this->log_output("Show Dlg Command sent");
		process[idx].write("PCOMMAND_SHOWDLG\n");
		break;
	}    

	return;
}