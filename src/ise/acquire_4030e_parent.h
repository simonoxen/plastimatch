/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _acquire_4030e_parent_h_
#define _acquire_4030e_parent_h_
#include "ise_config.h"
#include <QApplication>
#include <QProcess>
#include "acquire_4030e_window.h"
#include <fstream>
#include "acquire_4030e_define.h"


#define DEFAULT_LOGFILE_PATH "C:\\"

//added by YKP
#define MAX_PANEL_NUM_YK 2
#define MAX_RESEND_TRY 3

class Advantech;
class QTimer;

//class acquire_4030e_window;
//class Acquire_4030e_DlgControl;
//class QSystemSemaphore;

class Acquire_4030e_parent : public QApplication
{
    Q_OBJECT
    ;
public:
    Acquire_4030e_parent (int argc, char* argv[]);
    ~Acquire_4030e_parent ();

public:
    void initialize (int argc, char* argv[]);
    void kill_rogue_processes ();
    void log_output (const QString& log);
public slots:
    void timer_event ();
    //void timerSysTray_event();
    void timerCommandToChild0_event ();
	void timerCommandToChild1_event ();	
	void timerAboutToQuit_event ();
    void poll_child_messages ();
    void about_to_quit ();		
	//void FinalQuit();


public:
    Acquire_4030e_window *window;
    //Acquire_4030e_window *window2;
    //Acquire_4030e_DlgControl *m_dlgControl_0;
    //Acquire_4030e_DlgControl *m_dlgControl_1;
     int test;


    int num_process;
    QProcess process[2];
    QTimer *timer;
    QTimer *timerCommandToChild[2];
	QTimer *timerAboutToQuit; //for safe exit including child processes
    //QTimer *timerSysTray; //for periodic System Tray Showing
    Advantech *advantech;
    //bool generator_prep;
    Generator_state generator_state;
    bool panel_select;
    int panel_timer;

    PSTAT m_enPanelStatus[2]; //	init (ready), acquiring (yellow), ready (green)

    CommandToChild m_enLastRunnedStatus[2];

    QString m_program[2];
    QStringList m_arguments[2];

    //Destructor will not be called
    //In destructor: standby -> close link -> memory release
    bool RestartChildProcess(int idx);   

    void UpdateLableStatus();

    bool m_bPleoraErrorHasBeenOccurredFlag[2];
    //void GetPanelInfo(int idx); //Msg box

    void SendCommandToChild(int idx, CommandToChild enCommand); //Msg box    
    
    bool m_bWaitingForChildResponse[2]; //true: dont' send dummy msg
    CommandToChild m_enPrevSentCommand[2];
    //int m_iPrevSentPanelIdx;
    
    QString m_strLogFilePath;

    void StartCommandTimer(int idx, CommandToChild enCommand);
    int m_iMaxResendTryCnt[2];

    std::ofstream m_logFout;
	//void quit();//for reimplement

	bool m_bChildReadyToQuit[2]; //flag for child process 

	//bool m_bBusyParent;	

	bool m_bActivationHasBeenSent[2];

	bool m_bPanelRelayOpen0;
	bool m_bPanelRelayOpen1;


	//QSystemSemaphore* m_pSysSemaphore;


    //void ShowPanelControlWindow (int idx);   
    //bool SetStatus(int panelIdx, acquire_4030e_window::);
};
#endif
