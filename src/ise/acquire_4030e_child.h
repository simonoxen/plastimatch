/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _acquire_4030e_child_h_
#define _acquire_4030e_child_h_

#include <QApplication>
#include <QString>
#include <QCloseEvent>
//#include "HcpErrors.h"
//#include "HcpFuncDefs.h"
//#include "HcpSundries.h"
//#include <stdio.h>

typedef unsigned short USHORT;

#define MAX_CHECK_LINK_RETRY 3
#define RESTART_NEEDED -101
#define EXTERNAL_STATUS_CHANGE -102
#define DELAY_FOR_CHILD_RESPONSE 300

class Dips_panel;
class QTimer;
class Varian_4030e;
class Acquire_4030e_DlgControl;

class YK16GrayImage;
class DlgProgBarYK;

union UQueryProgInfo;
struct SModeInfo;

enum PSTAT{	
	NOT_OPENNED,
	OPENNED, //after openning, go to select receptor, vip_io_enable(active)
	PANEL_ACTIVE,	
	READY_FOR_PULSE,//print "ERADY for X-ray and go to wait-on-num-pulses
	PULSE_CHANGE_DETECTED, //beam signal detected
	IMAGE_ACQUSITION_DONE,	
	COMPLETE_SIGNAL_DETECTED,	
	DUMMY	
	//between every step, polling message will be runned,
	//especially in standby while loop, polling message always runs.
};


class Acquire_4030e_child : public QApplication
{
    Q_OBJECT
    ;
	//public signal:    
   
signals:
	void ready();

public slots:	
	//void About_To_Quit();
	//void m_timerPollMsg_event();
	//void pollMessageFromParent();
	void TimerPollMsgFromParent_event();
	//void TimerGetCrntPanelStatus_event();
	void TimerMainLoop_event(); //subtitute of large while loop in run func.
	//void timer_event ();
	//void test();
	//void TimerStart();

public:
    Acquire_4030e_child (int argc, char* argv[]);
    ~Acquire_4030e_child ();
public:
    bool init(const char* strProcNum, const char* strRecepterPath);
    int open_receptor (const char* path); //return value = result
    bool close_receptor ();
  

public:
    bool m_bNewMsgFromParentReady;
    PSTAT m_enPanelStatus;
    //bool m_bSoftHandshakingEnable; //true = software handshaking mode, false: hardware handshaking mode
    //bool m_bSoftBeamOn;
    //bool m_bSoftHandshakingEnableRequested; //new requested flag


    int idx;
    //QTimer *timerChild;
    Dips_panel *dp;
    Varian_4030e *vp;

    bool m_bAcquisitionOK; //determine "while" loop go or stop.    
    QString m_strProcNum;
    QString m_strReceptorPath;

    bool m_bPleoraErrorHasBeenOccurred;
    QString m_strFromParent;

    void InterpretAndFollow();    

    bool SWSingleAcquisition(UQueryProgInfo& crntStatus);    
    bool PerformDarkFieldCalibration(UQueryProgInfo& crntStatus, int avgFrameCnt);

    void PrintVoltage();

    //int m_iDarkImageCnt; //number of images for averaging (dark field and flood field)
    //USHORT** m_pAvgImageArr; //double pointer for buffer averaging
    int m_iGainImageCnt;

	YK16GrayImage* m_pCurrImage;
	YK16GrayImage* m_pDarkImage;
	YK16GrayImage* m_pGainImage;
    //USHORT* m_pDarkImage; //offset image buffer
    //USHORT* m_pGainImage; //Gain (flood or flat) image buffer

    void ReleaseMemory();

    bool ImageExportToDips(int sizeX, int sizeY, USHORT* pImg);
    //bool ImageExportToRawFile(const char* filePath, int sizeX, int sizeY, USHORT* pImg);
    bool AuditImage(int sizeX, int sizeY, USHORT* pImg);

	bool LoadDarkImage(QString& filePath);
	bool LoadGainImage(QString& filePath);

	QTimer * m_TimerPollMsgFromParent;	
	QTimer * m_TimerMainLoop;
	//QTimer * m_TimerGetCrntPanelStatus;

	UQueryProgInfo* m_pCrntStatus; //currentStatus    
	SModeInfo* m_pModeInfo;


	int m_timeOutCnt;

	///* Panel control functions // 03/06/2013
	bool PC_ReOpenPanel();
	bool PC_ActivatePanel();
	bool PC_WaitForPanelReady();
	bool PC_WaitForPulse();
	bool PC_GetImageHardware();
	bool PC_WaitForComplete();
	bool PC_ReStandbyPanel();


	bool PC_SoftwareAcquisition_SingleShot(); //should be done at READY_FOR_PULSE
	bool PC_DarkFieldAcquisition_SingleShot(int avgFrames); //should be done at READY_FOR_PULSE

	int m_iNumOfFramesRequested;// = 1 default
	Acquire_4030e_DlgControl *m_dlgControl;

	void ReDraw(int lowerWinVal, int upperWinVal);//current image only
	int m_iCurWinMidVal, m_iCurWinWidthVal; //window level of current image

	DlgProgBarYK* m_dlgProgBar;	

};

#endif