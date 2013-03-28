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
#include <QString>

class SleeperThread : public QThread {
public:
	static void msleep(unsigned long msecs) {
		QThread::msleep(msecs);
	}
};

//----------------------------------------------------------------------
//  main
//----------------------------------------------------------------------

#include <time.h>
#include <windows.h>
#include <iostream>
#include <fstream>
#include <QString>
//
//#if _WIN32_WINNT < 0x0501
#define MAX_LINE_LENGTH 512

bool gShowConsole;
//QString gStrLogoutput;//global log

using namespace std;

//HWND WINAPI GetConsoleWindowYK()
//{
//	HWND result;
//	char save_title[ 1000 ];
//	char temp_title[ 50 ];
//	time_t t = time( NULL );
//	lstrcpyA( temp_title, "test" );
//	lstrcatA( temp_title, ctime( &t ) );
//
//	GetConsoleTitleA( save_title, sizeof( save_title ) );
//	SetConsoleTitleA( temp_title );
//	result = FindWindowA( NULL, temp_title );
//	SetConsoleTitleA( save_title );
//
//	return result;
//}
//#endif 


int 
main (int argc, char* argv[])
{
	// The string such as "A422-07" is the imager serial number

	std::string default_path_1;
	std::string default_path_2;
	gShowConsole = true;

	//find config file    

	//HWND consoleW = GetConsoleWindowYK();
	//char str[200];
	//::GetWindowText(consoleW, LPSTR(str), 200);

	//  std::cout << "YK Log " << str << std::endl;

	//CloseWindow(consoleW);

	Acquire_4030e_child *child[2];
	child[0] = NULL;
	child[1] = NULL;	

	/* During debugging, use hard-coded path */
	if (argc < 2) { //when running without arguments

		ifstream fin;
		fin.open("acquire_4030e_config.txt"); //find it in same folder

		if (fin.fail())
		{
			cout << "acquire_4030e_config.txt file should exist in same folder" << endl;
			cout << "program will be terminated" << endl;
			exit(0);
		}

		char str[MAX_LINE_LENGTH];
		memset (str, 0, MAX_LINE_LENGTH);
		QString tmpReadString;

		fin.getline(str, MAX_LINE_LENGTH); // for header
		tmpReadString = str;

		if (!tmpReadString.contains("CONFIG_FILE_FOR_ACQUIRE4030E"))
		{
			cout <<"Un-proper config file! Program will be terminated!" << endl;
			exit(0);
		}

		while(!fin.eof())
		{
			memset (str, 0, MAX_LINE_LENGTH);
			fin.getline(str, MAX_LINE_LENGTH);
			tmpReadString = str;

			if (tmpReadString.contains("SHOW_CONSOLE_WINDOW"))
			{
				//parsing
				QStringList list = tmpReadString.split("\t");		
				if (list.count() == 2)
				{
					QString strConsoleTag = list.at(1);
					if (strConsoleTag.contains("0"))
					{
						gShowConsole = false;
					}
					else
					{
						gShowConsole = true;
					}
				}
			}

			else if (tmpReadString.contains("$PANEL_0_PATH"))
			{
				//parsing
				QStringList list = tmpReadString.split("\t");		
				if (list.count() == 2)
				{
					//QString tmp1 = list.at(1);
					//std::string test1 = tmp1.toStdString();
					default_path_1 = list.at(1).toStdString().c_str();
					//default_path_1 = list.at(1).toStdString().c_str();
				}
			}
			else if (tmpReadString.contains("$PANEL_1_PATH"))
			{
				QStringList list = tmpReadString.split("\t");		
				if (list.count() == 2)
				{		    
					default_path_2 = list.at(1).toStdString().c_str();
				}
			}
		}
		fin.close();

		char** argv_tmp = (char**) malloc (3 * sizeof(char*));
		argv_tmp[0] = argv[0];
		argv_tmp[1] = (char*) default_path_1.c_str();
		argv_tmp[2] = (char*) default_path_2.c_str();
		argv = argv_tmp;
		//argc = 2;
		argc = 3;
	}

	if (argc > 1) {
		if (!strcmp (argv[1], "--child")) {
			/*** Child process ***/
			printf ("A child is born.\n");		

			int panel_ldx = atoi(argv[3]);

			//arguments << "--child" << QString("%1").arg(i).toUtf8() << paths[i];
			//with no exception, argument number is 3:
			//argv[0] = process path
			//argv[1] = "--child"
			//argv[2] = no. of process
			//argv[3] = path			          

			if (panel_ldx < 2)
			{
				child[panel_ldx]= new Acquire_4030e_child (argc, argv);			

				if (!child[panel_ldx]->init(argv[2], argv[3])) //1) DP creation. if fail --> death of program //2) open receptor
				{
					//fail = receptor problem.
					//printf ("Receptor [%s] open failure. Check receptor cables.\n", argv[3]);		
					delete child[panel_ldx];
					child[panel_ldx] = NULL;
					//YK: 	 //when destructor activated, 	//Safe restart will be called
				}	    
				else
				{
					/* Wait forever */
					//printf ("Waiting forever (child).Before child->run\n");
					//child[panel_ldx]->run(); //start while loop.
					child[panel_ldx]->exec(); //start event handling
					//printf ("Wait complete (child).After child->run\n");
				}
			}			

#if defined (commentout)
			for (int i = 0; i < 10; i++) {
				printf ("A child is born.\n");
				fflush (stdout);
				SleeperThread::msleep(1000);
			}
#endif
		}
		else
		{
			/*** Parent process ***/

			/* Spawn child processes */
			Acquire_4030e_parent *parent //QCoreApplication		
				= new Acquire_4030e_parent (argc, argv);

			/* Wait forever */
			printf ("Waiting forever.(parent)\n");
			parent->exec ();
			printf ("Wait complete.(parent)\n");
			//after event, before destroy parent

			if (child[0] != NULL) //Here, child is NULL!!
			{
				child[0]->quit();
				delete child[0];				
				child[0] = NULL;
				

			}
			if (child[1] != NULL)
			{
				child[1]->quit();
				delete child[1];				
				child[1] = NULL;
			}
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
