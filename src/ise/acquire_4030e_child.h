/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _acquire_4030e_child_h_
#define _acquire_4030e_child_h_

#include <QApplication>
#include <QString>
#include <QCloseEvent>
#include <QLocalSocket>

#include <fstream>
#include "acquire_4030e_define.h"
//#include "HcpErrors.h"
//#include "HcpFuncDefs.h"
//#include "HcpSundries.h"
//#include <stdio.h>


class Dips_panel;
class QTimer;
class Varian_4030e;
class Acquire_4030e_DlgControl;

class YK16GrayImage;
//class DlgProgBarYK;
class QProgressDialog;

union UQueryProgInfo;
struct SModeInfo;

//class QLocalSocket;

//class QSystemSemaphore;


class Acquire_4030e_child : public QApplication
{
    Q_OBJECT
    ;
	//public signal:    
   
signals:
	void ready();

public slots:		
	//void TimerPollMsgFromParent_event();	
	void TimerMainLoop_event(); //subtitute of large while loop in run func.
	void SOCKET_ConnectToServer(QString& strServerName);	
	void SOCKET_ReadMessageFromParent(); //when msg comes to client
	void SOCKET_PrintError(QLocalSocket::LocalSocketError e);

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

	//QTimer * m_TimerPollMsgFromParent;	
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
	//bool PC_ReStandbyPanel();
	bool PC_WaitForStanby();


	bool PC_SoftwareAcquisition_SingleShot(); //should be done at READY_FOR_PULSE
	bool PC_DarkFieldAcquisition_SingleShot(int avgFrames); //should be done at READY_FOR_PULSE

	bool GetGainImageFromCurrent();	

	int m_iNumOfFramesRequested;// = 1 default
	Acquire_4030e_DlgControl *m_dlgControl;

	void ReDraw(int lowerWinVal, int upperWinVal);//current image only
	int m_iCurWinMidVal, m_iCurWinWidthVal; //window level of current image
	
	QProgressDialog* m_dlgProgBar;	

	//std::ofstream m_ImageInfoFout;	
	//QSystemSemaphore* m_pSysSemaphore;

	QLocalSocket* m_pClient;


};

#endif