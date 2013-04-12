/* -----------------------------------------------------------------------
See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <QTimer>
#include <QTime>
#include <QMessageBox>
#include <QDebug>
#include <math.h>

#include "acquire_4030e_child.h"
#include "aqprintf.h"
#include "dips_panel.h"
#include "varian_4030e.h"
#include "iostatus.h"
#include "HcpErrors.h"
#include "HcpFuncDefs.h"
#include "acquire_4030e_DlgControl.h"
#include "YK16GrayImage.h"
#include <QProgressDialog>
#include <QFileInfo>
#include <QDate>


#define TIMEOUT_MAINLOOP 50

Acquire_4030e_child::Acquire_4030e_child (int argc, char* argv[])
: QApplication (argc, argv)
{
	if (argc != 3) //1: exe path, 2: -- child 3: idx
		exit(-1);		
	
	m_iProcNum = atoi(argv[2]);

	dp = NULL;
	vp = NULL;

	m_bAcquisitionOK = false;
	

	m_bPleoraErrorHasBeenOccurred = false;

	//connect (this, SIGNAL(aboutToQuit()),this,SLOT(About_To_Quit())); //does not work

	m_enPanelStatus = NOT_OPENNED;

	//m_bSoftHandshakingEnable = false;

	//m_bSoftBeamOn = false;
	//m_bSoftHandshakingEnableRequested = false;

	/*aqprintf ("Child %s got request to open panel: %s\n", argv[2], argv[3]);
	this->idx = QString(argv[2]).toInt();
	this->dp = new Dips_panel;
	dp->open_panel (this->idx, HIRES_IMAGE_HEIGHT, HIRES_IMAGE_WIDTH);
	aqprintf ("DIPS shared memory created.\n");
	int result = this->open_receptor (argv[3]);    */

	//m_iDarkImageCnt = 8;//number of images for averaging (dark field and flood field)
	m_iGainImageCnt = 1;	

	m_pDarkImage = NULL;
	m_pGainImage = NULL;
	m_pCurrImage = NULL;

	m_pCurrImageRaw = NULL;
	m_pCurrImageDarkCorrected = NULL;


	//m_TimerPollMsgFromParent = new QTimer(this);
	m_TimerMainLoop = new QTimer(this);

	//connect(m_TimerPollMsgFromParent, SIGNAL(timeout()), this, SLOT(TimerPollMsgFromParent_event()));
	connect(m_TimerMainLoop, SIGNAL(timeout()), this, SLOT(TimerMainLoop_event()));


	m_dlgControl = new Acquire_4030e_DlgControl();
	m_dlgControl->setWindowTitle(QString("Panel Controller %1").arg(this->m_iProcNum));
	//m_dlgProgBar = new DlgProgBarYK((QWidget*)m_dlgControl);
	m_dlgProgBar = new QProgressDialog("Operation in progress.", "Cancel", 0, 100);
	m_dlgProgBar->setWindowModality(Qt::WindowModal);
	//connect(m_dlgProgBar, SIGNAL(canceled()), this, SLOT(ProgressCanceled());

	m_timeOutCnt = 0;
	m_iNumOfFramesRequested = 1;

	m_pCrntStatus = NULL;
	m_pModeInfo = NULL;	

	//m_iCurWinMidVal = DEFAULT_WINLEVEL_MID;
	//m_iCurWinWidthVal= DEFAULT_WINLEVEL_WIDTH;


	m_pClient = new QLocalSocket(this);
	connect(m_pClient, SIGNAL(readyRead()), this, SLOT(SOCKET_ReadMessageFromParent()));
	connect(m_pClient, SIGNAL(error(QLocalSocket::LocalSocketError)), this, SLOT(SOCKET_PrintError(QLocalSocket::LocalSocketError)));
	connect(m_pClient, SIGNAL(disconnected()),m_pClient, SLOT(deleteLater()));		



}

void Acquire_4030e_child::SOCKET_PrintError(QLocalSocket::LocalSocketError e)
{
	aqprintf("Error occurred in Socket communication, Error code = %d", e);
}

Acquire_4030e_child::~Acquire_4030e_child () // How to call this????
{
	vip_close_link();


	printf("Panel link is being closed..\n");    


	if (this->dp != NULL) {
		delete this->dp;
	}
	if (this->vp != NULL) {
		delete this->vp;
	}    


	ReleaseMemory();

	if (m_dlgControl != NULL)
	{
		delete m_dlgControl;
		m_dlgControl = NULL;
	}

	m_pClient->disconnectFromServer();
	delete m_pClient;

}

void Acquire_4030e_child::ReleaseMemory() //Dark or Gain Image buffer for correction
{	
	if (m_pCurrImage!= NULL)
	{		
		delete m_pCurrImage; //include Release Buffer 	
		m_pCurrImage = NULL;
	}
	if (!m_pDarkImage!= NULL)
	{	
		delete m_pDarkImage;	
		m_pDarkImage = NULL;
	}
	if (!m_pGainImage != NULL)
	{	
		delete m_pGainImage;	
		m_pGainImage = NULL;
	}
	if (!m_pCurrImageRaw!= NULL)
	{	
		delete m_pCurrImageRaw;	
		m_pCurrImageRaw = NULL;
	}
	if (!m_pCurrImageDarkCorrected!= NULL)
	{	
		delete m_pCurrImageDarkCorrected;	
		m_pCurrImageDarkCorrected = NULL;
	}	

	if (m_pCrntStatus != NULL)
	{	
		delete m_pCrntStatus;; //currentStatus    
		m_pCrntStatus = NULL;

	}
	if (m_pModeInfo == NULL)
	{	
		delete m_pModeInfo;	
		m_pModeInfo = NULL;
	}
	
	//delete m_pSysSemaphore;

	return;
}

//this func is called by main first, then by parent->RestartChild.
//if there is error during init, delete child will be called and so would be destructor (safe deletion)
//But, restart of parent will forcely shut down the process without calling destructor
//Destructor is important because of the closing panel
bool
//Acquire_4030e_child::init(const char* strProcNum, const char* strRecepterPath)
Acquire_4030e_child::init(int panelIdx) //procNum = panelIdx
{  	
	if (m_iProcNum != panelIdx)
	{
		aqprintf("Panel index is not correct_Init!\n");
		return false;
	}

	if (m_iProcNum != 0 && m_iProcNum != 1)
	{
		aqprintf("Panel index is not correct!\n");
		return false;
	}

	
	
	m_OptionSettingChild.CheckAndLoadOptions_Child(m_iProcNum);	

	QString strRecepterPath = m_OptionSettingChild.m_strDriverFolder[m_iProcNum];

	if (strRecepterPath.length() < 2)
	{
		aqprintf("Panel folder is not selected!\n");
		return false;
	}

	

	if (vp != NULL)
		vip_close_link(); //for sure, is it working??
	/*if (this->dp != NULL) {
	delete this->dp;
	dp = NULL;
	}*/
	if (this->vp != NULL) {
		delete this->vp;
		vp = NULL;
	}	
	m_iCurWinMidVal = DEFAULT_WINLEVEL_MID;
	m_iCurWinWidthVal= DEFAULT_WINLEVEL_WIDTH;
	
	ReleaseMemory(); // for m_pDarkImage and pGain	

	//START optionSetting

	m_pCrntStatus = new UQueryProgInfo; //currentStatus    		
	m_pModeInfo = new SModeInfo;		

	m_pDarkImage = new YK16GrayImage(); //offset image buffer
	m_pGainImage = new YK16GrayImage(); //Gain (flood or flat) image buffer
	m_pCurrImage = new YK16GrayImage();

	m_pCurrImageDarkCorrected = new YK16GrayImage();
	m_pCurrImageRaw = new YK16GrayImage();

	aqprintf ("Child %d got request to open panel: %s\n", m_iProcNum, strRecepterPath.toLocal8Bit().constData());

	//this->idx = QString(strProcNum).toInt();   
	this->idx = m_iProcNum;	

	//this->m_strReceptorPath = strRecepterPath;

	//if (dp != NULL)
	//	aqprintf("dp is not NULL\n");

	//Only do it once at first init.
	if (dp == NULL)
	{
		this->dp = new Dips_panel;
		dp->open_panel (this->idx, HIRES_IMAGE_HEIGHT, HIRES_IMAGE_WIDTH); //if failed, exit(1) -->very rare.    
		aqprintf ("DIPS shared memory created.\n");    
	}

	int result = open_receptor (strRecepterPath.toLocal8Bit().constData());	

	if (result == RESTART_NEEDED)
	{	
		aqprintf("RESTART_PROCESS\n"); //this will make parent to restart this process
		Sleep(2000);	

		return false; //if it is false, main will quit the process..
	}


	Sleep(100);
	vp->get_mode_info (*m_pModeInfo, vp->current_mode);

	//make image buffer
	if (vp->m_iSizeX*vp->m_iSizeY == 0)
	{
		aqprintf("YK Critical error! image size is not valid\n");
		return false;
	}

	//result = vip_reset_state();//0328 added. infinte loop of A584-09 panel.

	//this->m_pDarkImage = new USHORT [vp->m_iSizeX * vp->m_iSizeY];
	m_pCurrImage->CreateImage(vp->m_iSizeX , vp->m_iSizeY,0);
	m_pDarkImage->CreateImage(vp->m_iSizeX , vp->m_iSizeY,0);
	m_pGainImage->CreateImage(vp->m_iSizeX , vp->m_iSizeY,0);

	m_pCurrImageRaw->CreateImage(vp->m_iSizeX , vp->m_iSizeY,0);
	m_pCurrImageDarkCorrected->CreateImage(vp->m_iSizeX , vp->m_iSizeY,100);
	//aqprintf("m_pCurrImageDarkCorrected created\n");

		
	//this->m_pGainImage = new USHORT [vp->m_iSizeX * vp->m_iSizeY];



	//YK: It is mandatory!!!
	//vp->m_bDarkCorrApply = false;
	//vp->m_bGainCorrApply = false;

	//Sleep(3000);
	//Connect server
	//Sleep(1000);
	QString strServerName = QString("SOCKET_MSG_TO_CHILD_%1").arg(idx);	
	SOCKET_ConnectToServer(strServerName);
	//Sleep(1000);


	m_bLockInPanelStandby = true;
	m_bCancelAcqRequest = false;

	//this->m_bAcquisitionOK = true; 
	m_TimerMainLoop->start(100);
	ChangePanelStatus(IMAGE_ACQUSITION_DONE);	
	//aqprintf("SEVERNAME ERROR?\n");
	//aqprintf(strServerName.toLocal8Bit().constData());

	//SOCKET_ConnectToServer(strServerName);
	//Sleep(1000);	

	
	//aqprintf("about to call UpdateGUIFromSetting_Child");
	m_dlgControl->UpdateGUIFromSetting_Child();	//include reload dark/flood images	

	return true;
}

void Acquire_4030e_child::SOCKET_ConnectToServer(QString& strServerName)
{
	m_pClient->abort();
	m_pClient->connectToServer(strServerName);
	//aqprintf("SYSTEM_REQUEST_FOR_SERVER_CONNECTION\n");
}

void Acquire_4030e_child::SOCKET_ReadMessageFromParent() //when msg comes to client
{
	while(m_pClient->bytesAvailable()<(int)sizeof(quint32))
	{
		m_pClient->waitForReadyRead();
	}	

	QDataStream in(m_pClient);
	in.setVersion(QDataStream::Qt_4_0);
	if (m_pClient->bytesAvailable() < (int)sizeof(quint16))
	{
		return;
	}
	QString message;
	in >> message;

	
	m_strFromParent = message;

	//Response Msg
	//message.prepend("SYSTEM_");
	//message.append("\n");
	//aqprintf(message.toLocal8Bit().constData());

	InterpretAndFollow();
}

bool Acquire_4030e_child::close_receptor () //almost destructor...
{
	//aqprintf("Here is Close receptor");
	//if (vp == NULL)
	//	return false; //not expected close link

	if (vp != NULL)
	{
		vip_io_enable(HS_STANDBY);
		vip_close_link(); //for sure, is it working??	
		//aqprintf("link is closed\n");
	}

	if (this->vp != NULL) {
		delete this->vp;
		vp = NULL;
	}
	/*if (this->dp != NULL) {
	delete this->dp;
	dp = NULL;
	}*/
	ReleaseMemory();	

	return true;
}   

int Acquire_4030e_child::open_receptor (const char* path)
{
	//aqprintf ("Log by YKP1_path= [%d] %s\n", idx,path);
	//QMessageBox msgBox;

	QString tmpPath = path;
	if (tmpPath.length() < 1)
		return -1;

	
	//if there is no path exist, return false;    
	int result;
	//this->vp = new Varian_4030e (this->idx); //for each child vp will be assigned
	this->vp = new Varian_4030e (this->idx, this); //for each child vp will be assigned

	result = vp->open_link (idx, path); // idx of current child process		

	//if (result != HCP_NO_ERR)
	//	aqprintf("Error during open receptor. Error code = %d\n", result); //48 is 
	
	result = vp->disable_missing_corrections (result); //auto error correction precedure? 
	if (result != HCP_NO_ERR) {
		aqprintf ("vp.open_receptor_link returns error (%d): %s\n", result, Varian_4030e::error_string(result));
		//PrintCurrentTime();

		result = RESTART_NEEDED;		

		delete vp;
		vp = NULL;
		return result;        
	}
	Sleep(3000);//YK: after Open Receptor, give enough time (only once after running)   

	result = vp->check_link (MAX_CHECK_LINK_RETRY); //link checking, maximum 3 trials then re open the port	

	if (result == RESTART_NEEDED) //-111
	{
		aqprintf("Check_link failed.. process will be restarted!\n");	
		return result;
	}
	if (result != HCP_NO_ERR) {
		aqprintf ("vp.check_link returns error %d\n", result);
		vip_close_link();

		//msgBox.setText("test6");
		//msgBox.exec();

		delete vp;
		vp = NULL;
		return result;        
	}

	vp->print_sys_info ();

	//aqprintf ("Just Before select mode %d \n", vp->current_mode);
	//vp->current_mode is 0
	result = vip_select_mode (vp->current_mode);
	//aqprintf ("Just After select mode %d \n", vp->current_mode);
	vp->print_mode_info ();

	//aqprintf ("After select mode \n");

	if (result != HCP_NO_ERR) {
		aqprintf ("vip_select_mode(%d) returns error %d\n",vp->current_mode, result);	

		vp->close_link ();

		delete vp;
		vp = NULL;

		return result;
	}

	/* Spawn the timer for polling devices */
	//this->timerChild = new QTimer(this);
	//connect (timerChild, SIGNAL(timeout()), this, SLOT(timer_event()));
	//connect (this, SIGNAL(ready()), this, SLOT(pollMessageFromParent()));
	//timerChild->start(200);
	//aqprintf("YK: Timer would be started \n");   

	//after successfully openning the receptor,
	m_bPleoraErrorHasBeenOccurred = false;
	//m_enPanelStatus = COMPLETE_SIGNAL_DETECTED;  // go to standby
	//this->m_timeOutCnt = 0;
	//this->m_bAcquisitionOK = true;

	//ChangePanelStatus(IMAGE_ACQUSITION_DONE);

	return result;
}

//Acquire N frames and average them to make a one raw image file and to memory
bool Acquire_4030e_child::PerformDarkFieldCalibration(UQueryProgInfo& crntStatus, int avgFrameCnt)
{   
	// if (vp->m_bDarkCorrApply)
	//aqprintf("m_bDarkCorrApply is true\n");

	if (avgFrameCnt<1 ||avgFrameCnt > 30)
	{
		aqprintf("Invalid frame number for averaging. Should be < 9 and > 0\n");
		return false;
	}

	int result;
	int i = 0;

	//in DualRadTest, this code is not existing
	
	//Make buffer array

	//USHORT** tmpImageBufArr = NULL;

	int iWidth= vp->m_iSizeX;
	int iHeight= vp->m_iSizeY;
	int imgSize = iWidth*iHeight;

	if (imgSize == 0)
		return false;

	long* tmpImageSumBuf = NULL;
	tmpImageSumBuf = new long [imgSize];

	for (int j = 0 ; j<imgSize; j++)
	{
		tmpImageSumBuf[j] = 0;
	}
	int SumSuccessCnt = 0;


	//result = vip_io_enable(HS_STANDBY); //Should not be called before vip_enable_sw_handshaking(TRUE);!!! 04 / 12
	//m_timeOutCnt = 0;

	//while(result == HCP_NO_ERR && (crntStatus.qpi.NumFrames + crntStatus.qpi.ReadyForPulse + crntStatus.qpi.Complete + crntStatus.qpi.NumPulses > 0))		
	//{
	//	result = vp->query_prog_info (crntStatus);
	//	Sleep(500);

	//	m_timeOutCnt += TIMEOUT_MAINLOOP;

	//	if (m_timeOutCnt > 5000)
	//	{
	//		aqprintf("*** TIMEOUT ***Stuck in waiting for Standby signal.\n");							
	//		result = HCP_TIMEOUT;
	//		//break;
	//		return false;
	//	}			
	//}

	m_dlgProgBar->setValue(10);	

	//Repeat below procedure //MultiFrame
	for (i = 0 ; i< avgFrameCnt ; i++)
	//for (i = 0 ; i< 1 ; i++)
	{		
		result = HCP_NO_ERR;
		m_timeOutCnt = 0;

		//result = vip_reset_state(); //Mandatory!!! or reopen the panel

		aqprintf("Acquiring image number %d of total %d...\n", i+1, avgFrameCnt);

		result = vip_enable_sw_handshaking(TRUE);
		//1] START ACQUISITION
		if (result == HCP_NO_ERR)
		{
			result = vip_sw_handshaking(VIP_SW_PREPARE, TRUE);
			if (result == HCP_NO_ERR)
			{
				result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, TRUE);
				if (result == HCP_NO_ERR)
				{
					//aqprintf("YK3\n");
					crntStatus.qpi.Complete = FALSE; //YK: seems this is Important skill!!!
					crntStatus.qpi.ReadyForPulse = FALSE; //YK: not sure..

					result = vp->query_prog_info(crntStatus);

					//Sleep(100);

										
					while(result == HCP_NO_ERR && crntStatus.qpi.ReadyForPulse == FALSE)
					{
						result = vp->query_prog_info (crntStatus);
						Sleep(100);

						m_timeOutCnt += TIMEOUT_MAINLOOP;

						if (m_timeOutCnt > 3000)
						{
							aqprintf("*** TIMEOUT ***Stuck in waiting for ReadyForPulse signal. Image acquisition failed!\n");							
							result = HCP_TIMEOUT;
							break;
						}			
					}
					Sleep(1000); //original:500
				}
			}
		}		
		aqprintf("Receptor is enabled for acquisition\n");

		//2] WAIT AC
		m_timeOutCnt = 0;

		if (result == HCP_NO_ERR)
		{	    
			aqprintf("Wait for complete\n");
			result = vp->query_prog_info (crntStatus);
			Sleep(100);

			if (result == HCP_NO_ERR)
			{	
				
				while (crntStatus.qpi.Complete != 1 && result == HCP_NO_ERR) //YK TEMP: stuck here!!!!!!!!!!!!!!!!
				{
					result = vp->query_prog_info (crntStatus);
					Sleep(100);
					m_timeOutCnt += TIMEOUT_MAINLOOP;
					if (m_timeOutCnt > 3000)
					{
						aqprintf("*** TIMEOUT ***Stuck in waiting for complete signal. Image acquisition failed!\n");							
						result = HCP_TIMEOUT;
						break;
					}					
				}
			}
		}

		//aqprintf("About to Get image\n");

		if (result == HCP_NO_ERR)
		{	    	
			//aqprintf("GetImage\n");
			int mode_num = vp->current_mode;

			//int npixels = iWidth * iHeight;

			USHORT *image_ptr = (USHORT *)malloc(imgSize * sizeof(USHORT));

			result = vip_get_image(mode_num, VIP_CURRENT_IMAGE, iWidth, iHeight, image_ptr);			
			
			//result = vip_get_image(mode_num, imageType, xSize, ySize, tmpImageBufArr[i]);
			if(result != HCP_NO_ERR)
			{
				aqprintf("*** vip_get_image returned error %d\n", result);
			}
			else
			{
				if (AuditImage(iWidth, iHeight, image_ptr))
				{
					SumSuccessCnt++;
					for (int j = 0 ; j < imgSize; j++)
					{
						tmpImageSumBuf[j] = tmpImageSumBuf[j] + (long)image_ptr[j];
					}
				}
			}
			free(image_ptr);
		}

		if (result == HCP_NO_ERR)
		{
			//aqprintf("VIP_SW_VALID_XRAYS\n");
			result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, FALSE);
			if (result != HCP_NO_ERR)
			{
				aqprintf("vip_sw_handshaking(VIP_SW_VALID_XRAYS, FALSE) returns %d\n", result);
			}
			result = vip_sw_handshaking(VIP_SW_PREPARE, FALSE);
			if (result != HCP_NO_ERR)
				aqprintf("vip_sw_handshaking(VIP_SW_PREPARE, FALSE) returns %d\n", result);

			if (i != avgFrameCnt-1) // in last term, don't do the hardware handshaking on --> will be done in OPENNED
			{
				result = vip_enable_sw_handshaking(FALSE);// YK0411
				if (result != HCP_NO_ERR)
					aqprintf("vip_enable_sw_handshaking(FALSE) returns %d\n", result);
			}			
		}
		//Repeat above procedure //MultiFrame

		int tmpVal = (int)(80.0 / (double)(avgFrameCnt) * (i+1)); // 10 + 80 + 10
		m_dlgProgBar->setValue(tmpVal);
		//aqprintf("Prog %d\n", tmpVal);
	} //end of for loop

	//in DualRadTest, this code is not existing
	

	//result = vip_reset_state();

	//aqprintf("Ready to export\n");

	//Save in memory
	if (!m_pDarkImage->IsEmpty())
	{
		for (int j = 0 ; j<imgSize ; j++)
		{
			m_pDarkImage->m_pData[j] = (USHORT)(tmpImageSumBuf[j]/(double)SumSuccessCnt);
		}
		aqprintf("%d frames were added to dark field for averaging. Dark image buffer is ready.\n", SumSuccessCnt);	
	}
	else
	{
		aqprintf("YK Critical error! Dark image buffer is not ready.\n");
	}
	delete [] tmpImageSumBuf;

	m_dlgProgBar->setValue(95);
	//aqprintf("Prog 95\n");


	int idx = this->m_iProcNum;
	//QString strFileName = QString("C:\\DarkImage_Avg%1").arg(SumSuccessCnt);
	QString strFolderName = m_OptionSettingChild.m_strDarkImageSavingFolder[idx];
	QString strFileName = QString("\\%1_Dark_Avg%2").arg(idx).arg(SumSuccessCnt);

	QDate date = QDate::currentDate();	
	QString strDate = date.toString("_yyyy_MM_dd");

	QTime time = QTime::currentTime();    
	QString strTime = time.toString("_hh_mm_ss");
	strFileName.append(strDate);
	strFileName.append(strTime);
	strFileName.append(".raw");

	strFileName.prepend(strFolderName);

	//Dark image no need to send to dips

	/*if (!ImageExportToDips(iWidth, iHeight, m_pDarkImage))
	{
	aqprintf("Cannot export to dips\n");
	}*/
	//if (!ImageExportToRawFile(strFileName.toLocal8Bit().constData(), iWidth, iHeight, m_pDarkImage))
	if (!m_pDarkImage->SaveDataAsRaw(strFileName.toLocal8Bit().constData()))
	{
		aqprintf("Cannot export to raw file\n");
	}

	//UpdateDarkImagePath
	m_dlgControl->lineEditDarkPath->setText(strFileName);

	//if (!m_pDarkImage->FillPixMap(m_iCurWinMidVal, m_iCurWinWidthVal)) //16 bit to 8 bit
	//if (!m_pDarkImage->FillPixMap(DEFAULT_WINLEVEL_MID, DEFAULT_WINLEVEL_WIDTH)) //16 bit to 8 bit
	if (!m_pDarkImage->FillPixMap(1500, 3000)) //0 - 3000 is enough
		aqprintf("Error on FillPixMap\n");

	if (!m_pDarkImage->DrawToLabel(this->m_dlgControl->lbDarkField)) //SetPixMap 
		aqprintf("Error on drawing");
		
	//result = vp->get_image_to_file(vp->m_iSizeX, vp->m_iSizeY, (char*)(strFileName.toLocal8Bit().constData()),VIP_CURRENT_IMAGE);
	//   if (vp->m_bDarkCorrApply)
	//	aqprintf("m_bDarkCorrApply is true\n");

	m_dlgProgBar->setValue(100);	
	//aqprintf("Prog 99\n");

	if (result != HCP_NO_ERR)
		return false;   

	return true;
}


bool Acquire_4030e_child::SWSingleAcquisition(UQueryProgInfo& crntStatus)
{ 		

	int result = HCP_NO_ERR;
	//in DualRadTest, this code is not existing
	result = vip_reset_state(); //Mandatory!!! or reopen the panel.    

	//aqprintf("Reset State Complete\n");
	m_timeOutCnt = 0;

	int maxCnt = 0;    
	result = vip_enable_sw_handshaking(TRUE);

	//aqprintf("SW handshaking enabled1\n");
	//1] START ACQUISITION
	if (result == HCP_NO_ERR)
	{
		result = vip_sw_handshaking(VIP_SW_PREPARE, TRUE);
		//aqprintf("SW handshaking enabled2\n");

		if (result == HCP_NO_ERR)
		{
			result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, TRUE);
			//aqprintf("SW handshaking enabled3\n");
			if (result == HCP_NO_ERR)
			{
				//aqprintf("YK3\n");
				crntStatus.qpi.Complete = FALSE; //YK: seems this is Important skill!!!
				crntStatus.qpi.ReadyForPulse = FALSE; //YK: not sure..

				result = vp->query_prog_info(crntStatus); //single trial --> OK 0 0 0 1

				Sleep(100);				

				m_dlgProgBar->setValue(25);

				while(result == HCP_NO_ERR && crntStatus.qpi.ReadyForPulse == FALSE)
				{				

					result = vp->query_prog_info (crntStatus);
					Sleep(100);
					/*maxCnt++;
					if (maxCnt > 1000)
					{
						break;
					}*/			
					m_timeOutCnt += TIMEOUT_MAINLOOP;
					if (m_timeOutCnt > 3000)
					{
						aqprintf("*** TIMEOUT ***Stuck in waiting ReadyForPulse. Image acquisition failed!\n");
						m_timeOutCnt = 0;
						return false;
					}					
				}
				
				Sleep(1000);
				m_dlgProgBar->setValue(50);
			}
		}
	}
	aqprintf("Receptor is enabled for acquisition\n");

	//2] WAIT AC
	m_timeOutCnt = 0;

	if (result == HCP_NO_ERR)
	{
		//aqprintf("YK4\n");
		result = vp->query_prog_info (crntStatus); // 1 0 1 1

		Sleep(100);		

		if (result == HCP_NO_ERR)
		{
			//aqprintf("YK4_2\n");
			//stuck in here
			while (crntStatus.qpi.Complete != 1 && result == HCP_NO_ERR)
			{
				result = vp->query_prog_info (crntStatus); //1 1 1 0
				Sleep(100);

				m_timeOutCnt += TIMEOUT_MAINLOOP;
				if (m_timeOutCnt > 3000)
				{
					aqprintf("*** TIMEOUT ***Stuck in waiting for Complete signal. Image acquisition failed!\n");
					m_timeOutCnt = 0;
					return false;
				}				
			}	    
		}
	}
	m_dlgProgBar->setValue(75);
	//3] GET IMAGE	

	//aqprintf(str.toLocal8Bit().constData());
	if (result == HCP_NO_ERR)
	{		
		//result = vp->get_image_to_dips (dp, vp->m_iSizeX, vp->m_iSizeY);
		//result = vp->get_image_to_file(vp->m_iSizeX, vp->m_iSizeY, (char*)(strFileName.toLocal8Bit().constData()),VIP_CURRENT_IMAGE);		
		result = vp->get_image_to_buf(vp->m_iSizeX, vp->m_iSizeY); //fill Curr Image

		vp->CopyFromBufAndSendToDips(dp);
			

		//Display on the screen
		m_pCurrImage->FillPixMap(m_iCurWinMidVal, m_iCurWinWidthVal);
		m_pCurrImage->DrawToLabel(m_dlgControl->lbCurrentImage);

		//Save as raw file
		if (m_dlgControl->ChkAutoSave->isChecked())
		{
			QString folderPath = m_dlgControl->lineEditCurImageSaveFolder->text();
			//QLineEdit* edit;
			//edit->text()

			if (folderPath.length() < 2)
				folderPath = "C:";

			QString strFileName = QString("%1_CurImg").arg(this->idx); //panel number display
			QDate date = QDate::currentDate();
			QTime time = QTime::currentTime();    
			QString strDate = date.toString("_yyyy_MM_dd");
			QString strTime = time.toString("_hh_mm_ss");        
			strFileName.append(strDate);
			strFileName.append(strTime);
			strFileName.append(".raw");

			strFileName.prepend("\\");
			strFileName.prepend(folderPath);
			m_pCurrImage->SaveDataAsRaw(strFileName.toStdString().c_str());	

		}
	}
	if (result == HCP_NO_ERR)
	{		
		aqprintf("VIP_SW_VALID_XRAYS\n");

		result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, FALSE);
		if (result != HCP_NO_ERR)
			aqprintf("vip_sw_handshaking(VIP_SW_VALID_XRAYS, FALSE) returns %d\n", result);
		result = vip_sw_handshaking(VIP_SW_PREPARE, FALSE);
		if (result != HCP_NO_ERR)
			aqprintf("vip_sw_handshaking(VIP_SW_PREPARE, FALSE) returns %d\n", result);
		result = vip_enable_sw_handshaking(FALSE);
		if (result != HCP_NO_ERR)
			aqprintf("vip_enable_sw_handshaking(FALSE) returns %d\n", result);
	}

	m_dlgProgBar->setValue(100);

	//in DualRadTest, this code is not existing
	
	Sleep(500);
	
	if (result != HCP_NO_ERR)
		return false;	
	

	return true;
}

//To avoid aquiring dark field image during proton-beam or X-ray on
//Judge the validity of dark field by using histogram or mean value or SD
bool Acquire_4030e_child::AuditImage(int sizeX, int sizeY, USHORT* pImg)
{  
	int nTotal;
	long minPixel, maxPixel;
	int i;
	double pixel, sumPixel;

	int npixels = sizeX*sizeY;
	nTotal = 0;
	//minPixel = 4095;
	minPixel = 65535;
	maxPixel = 0;
	sumPixel = 0.0;

	for (i = 0; i < npixels; i++)
	{
		pixel = (double) pImg[i];
		sumPixel += pixel;
		if (pImg[i] > maxPixel)
			maxPixel = pImg[i];
		if (pImg[i] < minPixel)
			minPixel = pImg[i];
		nTotal++;
	}

	double meanPixelval = sumPixel / (double)nTotal;    

	double sqrSum = 0.0;
	for (i = 0; i < npixels; i++)
	{
		sqrSum = sqrSum + pow(((double)pImg[i] - meanPixelval),2.0);
	}
	double SD = sqrt(sqrSum/(double)nTotal);

	aqprintf("Mean=%4.3f SD=%4.3f \n",meanPixelval, SD);

	//YKTEMP: should be implemented
	double MeanUpper = 6000;
	double MeanLower = 0;
	double SDUpper = 3000;
	double SDLower = 0;
	//YKTEMP: should be implemented

	if (meanPixelval > MeanUpper || meanPixelval < MeanLower || 
		SD > SDUpper || SD < SDLower)
	{
		aqprintf("This image is not proper for Dark Image.MeanVal:%3.5f, SD:%3.5f \n", meanPixelval, SD);
		return false;	
	}
	return true;
}
bool Acquire_4030e_child::ImageExportToDips(int sizeX, int sizeY, USHORT* pImg)
{
	if (dp == NULL)
		return false;

	dp->wait_for_dips ();

	for (int i = 0; i < sizeX * sizeY; i++) {
		dp->pixelp[i] = pImg[i];
	}
	dp->send_image ();

	return true;
}

void Acquire_4030e_child::InterpretAndFollow ()
{
	//m_strFromParent.append("\n");
	//aqprintf(m_strFromParent.toLocal8Bit().constData()); //PCOMMAND_..
	
	if (m_strFromParent.length() < 1)
	{
		// do nothing
		return;
	}
	if (!m_strFromParent.contains("PCOMMAND"))
	{
		aqprintf("m_strFromParent does not contain PCOMMAND\n");
		return;
	}

	if (m_strFromParent.contains("SHOWDLG"))
	{
		//aqprintf("Show Dlg Command Received\n");
		//Sleep(DELAY_FOR_CHILD_RESPONSE);
		if (m_dlgControl != NULL)
			m_dlgControl->show();

		Sleep(DELAY_FOR_CHILD_RESPONSE);
		aqprintf("PRESPP_SHOWDLG\n");		
	}
	else if (m_strFromParent.contains("KILL"))//m_enPanelStatus to NoT openned
	{
		m_bAcquisitionOK = false;
		Sleep(DELAY_FOR_CHILD_RESPONSE);

		close_receptor(); //Like Destructor

		if (this->dp != NULL) {
			delete this->dp;
			dp = NULL;
		}
		m_enPanelStatus = NOT_OPENNED;
		this->m_timeOutCnt = 0;
		//m_TimerPollMsgFromParent->stop();
		m_TimerMainLoop->stop(); //if running
		aqprintf("PRESPP_KILL\n");
	}	
	else if (m_strFromParent.contains("UNLOCKFORPREPARE"))//msg is sent from parent's main loop related to advantech
	{		
		aqprintf("PRESPP_UNLOCKFORPREPARE\n");
		
		//m_TimerMainLoop->start(100); //from standby (0 0 0 0) --> hardwarehandshaking & HS_ACTIVE
		//m_bAcquisitionOK = true; //resume the main loop timer
		m_bLockInPanelStandby = false;
		m_bCancelAcqRequest = false; //initialization
	}	
	else if (m_strFromParent.contains("CANCELACQ"))//msg is sent from parent's main loop related to advantech
	{	
		//aqprintf("Now chaning m_bCancelAcqRequest stuats to true\n");
		m_bCancelAcqRequest = true;//Skip ReadyForPulse when the other panel is selected
		//m_bLockInPanelStandby= true;
		aqprintf("PRESPP_CANCELACQ\n");
	}	
	m_strFromParent="";
	return;
}

void Acquire_4030e_child::PrintVoltage()
{
	int  uType = HCP_U_QPIVOLTS;
	UQueryProgInfo  QueryProgressInfoData;

	memset(&QueryProgressInfoData.qpivolts, 0, sizeof(SQueryProgInfoVolts));
	QueryProgressInfoData.qpivolts.StructSize = sizeof(SQueryProgInfoVolts);

	int result = HCP_NO_ERR;

	result = vip_query_prog_info(uType, &QueryProgressInfoData);
	//aqprintf("YK: Result is %d\n",result);
	Sleep(100);

	// If the call was successful print out the voltages
	if (result == HCP_NO_ERR)
	{
		for (int i = 0; i < QueryProgressInfoData.qpivolts.NumSensors; i += 4)
		{
			aqprintf("YK: Volts(V) %2d..%-2d: %7.3f, %7.3f, %7.3f, %7.3f\n", i, i+3,
				QueryProgressInfoData.qpivolts.Volts[i], QueryProgressInfoData.qpivolts.Volts[i+1],
				QueryProgressInfoData.qpivolts.Volts[i+2], QueryProgressInfoData.qpivolts.Volts[i+3]);
		}
	}

}

void Acquire_4030e_child::TimerMainLoop_event() //called every 50 ms
{
	if (!m_bAcquisitionOK)
		return;
	//stop main loop to go into the sub loop
	//At the end of the sub-loop code, resume the main loop	
	int result = HCP_NO_ERR;
	QString errorStr;

	int iAvgFrames = 0;

	switch (m_enPanelStatus)
	{
	case NOT_OPENNED : //is not used in normal case
		m_bAcquisitionOK = false; //BUSY flag
		PC_ReOpenPanel();
		m_bAcquisitionOK = true;
		//m_TimerMainLoop->start(TIMEOUT_MAINLOOP)  //should be 100 ms
		break;		

	case OPENNED:	
		m_bAcquisitionOK = false; //BUSY flag
		PC_ActivatePanel();
		m_bAcquisitionOK = true; //BUSY flag		
		break;

	case PANEL_ACTIVE: 		
		m_bAcquisitionOK = false; //BUSY flag
		PC_WaitForPanelReady(); //will be repeated several times unitl the status change.
		m_bAcquisitionOK = true; //BUSY flag		
		
		break;

	case READY_FOR_PULSE: //Stand-by status			

		if (m_bCancelAcqRequest)//Skip ReadyForPulse when the other panel is selected
		{
			aqprintf("Status changed to IMAGE_ACQUSITION_DONE\n");
			ChangePanelStatus(IMAGE_ACQUSITION_DONE);			
			m_bCancelAcqRequest = false;
		}
		else
		{
			m_bAcquisitionOK = false; //BUSY flag		
			PC_WaitForPulse();			
			m_bAcquisitionOK = true; //BUSY flag
		}				
		break;

	case PULSE_CHANGE_DETECTED:
		m_bAcquisitionOK = false; //BUSY flag
		//PC_GetImageHardware();
		//m_enPanelStatus = IMAGE_ACQUSITION_DONE;
		PC_WaitForComplete(); //inside here image acquisition
		m_bAcquisitionOK = true; //BUSY flag
		break;            	    	

	case COMPLETE_SIGNAL_DETECTED: // Go back to first step  PC_ReStandbyPanel	
		m_bAcquisitionOK = false; //BUSY flag				
		PC_GetImageHardware(); //called once		
		m_bAcquisitionOK = true; //BUSY flag
		break;	

	case IMAGE_ACQUSITION_DONE: //once called
		m_bAcquisitionOK = false; //BUSY flag

		this->PC_CallForStanby();
		m_bLockInPanelStandby = true;
		m_bCancelAcqRequest = false;

		m_bAcquisitionOK = true; //BUSY flag	

		break;          

	case STANDBY_CALLED:
		m_bAcquisitionOK = false; //BUSY flag
		PC_WaitForStanby(); //inside here image acquisition
		m_bAcquisitionOK = true; //BUSY flag		
		break;

	case STANDBY_SIGNAL_DETECTED:
		//m_bAcquisitionOK = false; //Just hold the timer

		if (!m_bLockInPanelStandby)
		{			
			ChangePanelStatus(OPENNED);
		}
		break;

	case ACQUIRING_DARK_IMAGE:
		m_bAcquisitionOK = false;
		iAvgFrames = m_dlgControl->SpinDarkAvgFrames->value();	
		this->PC_DarkFieldAcquisition_SingleShot(iAvgFrames);
		ChangePanelStatus(OPENNED);	
		m_bAcquisitionOK = true;
		break;
	case DUMMY:		
		break;
	} //end of switch
}

//**************implementation of each function according to panel status *******************//
// PC means Panel Control functions group
bool Acquire_4030e_child::PC_ReOpenPanel()
{
	if (m_enPanelStatus != NOT_OPENNED)
	{
		aqprintf ("PC_ReOpenPanel Error: panel status is not proper\n");
		return false;
	}

	aqprintf ("PSTAT0: NOT_OPENNED\n");
	//if (!init(m_iProcNum, this->m_strReceptorPath.toLocal8Bit().constData()))
	if (!init(m_iProcNum))
	{			
		aqprintf ("Cannot Open the panel! Loop will be stopped.\n");
		m_bAcquisitionOK = false;		
	}
	else
	{
		ChangePanelStatus(OPENNED);
		/*m_enPanelStatus = OPENNED;	
		this->m_timeOutCnt = 0;*/
	}

	//aqprintf ("PSTAT1: OPENNED\n");
	return true;
}

// PC means Panel Control functions group
bool Acquire_4030e_child::PC_ActivatePanel()
{
	if (m_enPanelStatus != OPENNED)
	{
		aqprintf ("PC_ActivatePanel Error: panel status is not proper\n");
		return false;
	}
	//aqprintf ("PSTAT1: OPENNED\n");

	int result = HCP_NO_ERR; 

	result = vip_enable_sw_handshaking (FALSE); //mandatory code	

	if (result != HCP_NO_ERR)
	{
		aqprintf ("**** vip_enable_sw_handshaking returns error %d\n",result);		
		
		if (result == 2)//state error needs to RESTART
		{				
			aqprintf("RESTART_PROCESS\n"); //this will make parent to restart this process
			Sleep(2000);	
		}
		else
		{
			close_receptor();			
			m_enPanelStatus = NOT_OPENNED;
			this->m_timeOutCnt = 0;
			Sleep(300);
		}
		return false;
	}				
		
	result = vip_io_enable (HS_ACTIVE); // Frame | Complete | NumPulse | PanelReady		

	if (result != HCP_NO_ERR) {	
		aqprintf("**** returns error %d - Panel Activation Error\n", result);		
	}
	//m_enPanelStatus = PANEL_ACTIVE;	
	//a/qprintf ("PSTAT2: PANEL_ACTIVE\n");	

	ChangePanelStatus(PANEL_ACTIVE);

	m_timeOutCnt = 0;	
	return true;
}


// PC means Panel Control functions group
bool Acquire_4030e_child::PC_WaitForPanelReady()
{
	if (m_enPanelStatus != PANEL_ACTIVE)
	{
		aqprintf ("PC_ActivatePanel Error: panel status is not proper\n");
		return false;
	}

	m_pCrntStatus->qpi.ReadyForPulse = FALSE;
	
	int result = vp->query_prog_info (*m_pCrntStatus); // 0 0 0 0 --> 1 1 0 0	

	m_timeOutCnt += TIMEOUT_MAINLOOP;

	if (result != HCP_NO_ERR)
	{
		aqprintf("Error on querying_in PC_WaitForPanelReady with an error code of %d\n", result);			
		//vip_io_enable(HS_STANDBY); //should be!!
		//m_enPanelStatus = OPENNED; //Go back to first step		
		//this->m_timeOutCnt = 0;
		ChangePanelStatus(IMAGE_ACQUSITION_DONE);
	}
	else if (m_timeOutCnt > 5000) //Time_out //5000 is fixed! Optimized value
	{
		aqprintf("*** TIMEOUT ***_wait_on_ready_for_pulse\n"); //just retry
		
		aqprintf("frames=%d complete=%d pulses=%d ready=%d\n",
				m_pCrntStatus->qpi.NumFrames,
				m_pCrntStatus->qpi.Complete,
				m_pCrntStatus->qpi.NumPulses,
				m_pCrntStatus->qpi.ReadyForPulse);

	/*	m_timeOutCnt = 0;
		vip_io_enable(HS_STANDBY);
		Sleep(1000);
		m_enPanelStatus = OPENNED;*/

		ChangePanelStatus(IMAGE_ACQUSITION_DONE);
		return false;
	}
	else
	{
		if (m_pCrntStatus->qpi.ReadyForPulse == 1)
		{			
			/*m_enPanelStatus = READY_FOR_PULSE;
			aqprintf ("PSTAT3: READY_FOR_PULSE\n");
			Sleep(100);
			aqprintf("READY FOR X-RAYS - EXPOSE AT ANY TIME\n");
			m_timeOutCnt = 0;	*/	

			ChangePanelStatus(READY_FOR_PULSE);

		}
		else
		{
			//continue loop
		}
	}
	return true;
}


// PC means Panel Control functions group
bool Acquire_4030e_child::PC_WaitForPulse() //vp->wait_on_num_pulses 
{
	if (m_enPanelStatus != READY_FOR_PULSE)
	{
		aqprintf ("PC_WaitForPulse Error: panel status is not proper\n");
		return false;
	}

	int prevNumPulses = m_pCrntStatus->qpi.NumPulses;

	//m_pSysSemaphore->acquire();
	int result = vp->query_prog_info (*m_pCrntStatus);
	//m_pSysSemaphore->release(3);

	m_timeOutCnt += TIMEOUT_MAINLOOP;

	if (result != HCP_NO_ERR)
	{
		aqprintf("*** Acquisition terminated with error %d\n", result);
		aqprintf("Error on querying in PC_WaitForPulse with error code of %d\n", result);

		ChangePanelStatus(IMAGE_ACQUSITION_DONE);

		//m_enPanelStatus = OPENNED;//go back to first step	
		//vip_io_enable(HS_STANDBY);
		//this->m_timeOutCnt = 0;
		return false;
	}
	//else if (m_timeOutCnt > 5000) 
	//{		
	//	m_enPanelStatus = IMAGE_ACQUSITION_DONE; //GOTO Standby
	//	return false;
	//}
	if (m_pCrntStatus->qpi.ReadyForPulse == 0 && m_pCrntStatus->qpi.NumPulses == 0) //abnormal case
	{
		ChangePanelStatus(OPENNED);
	}
	else
	{
		if (m_pCrntStatus->qpi.NumPulses != prevNumPulses) //if thiere is any change		
		{
			ChangePanelStatus(PULSE_CHANGE_DETECTED);
			/*m_enPanelStatus = PULSE_CHANGE_DETECTED;
			aqprintf ("PSTAT4: PULSE_CHANGE_DETECTED\n");
			m_timeOutCnt = 0;*/
		}	
	}
	return true;
}

bool Acquire_4030e_child::PC_WaitForComplete()//vp->wait_on_complete
{
	if (m_enPanelStatus != PULSE_CHANGE_DETECTED)
	{
		aqprintf ("PC_WaitForComplete Error: panel status is not proper\n");
		m_enPanelStatus = OPENNED;
		return false;
	}	

	QString errorStr;
	int result = HCP_NO_ERR;
	m_pCrntStatus->qpi.Complete = FALSE;

	m_timeOutCnt += TIMEOUT_MAINLOOP;

	//m_pSysSemaphore->acquire();
	result = vp->query_prog_info (*m_pCrntStatus); // YK: Everytime, No data error 8
	//m_pSysSemaphore->release(3);

	if (result != HCP_NO_ERR) //Not that serious in this step
	{	
		errorStr = QString ("Error in wait_on_complete. Error code is %1. Now, retrying...\n").arg(result);		
		aqprintf(errorStr.toLocal8Bit().constData());

		ChangePanelStatus(IMAGE_ACQUSITION_DONE);

		//vip_io_enable(HS_STANDBY); // Let's try without this (directly go to PANEL_ACTIVE)
		//Sleep(1000);
		//m_enPanelStatus = OPENNED;//Let's try with PANEL_ACTIVE (without reselection and HS_ACTIVE setting		
		//this->m_timeOutCnt = 0;
		return false;
	}
	else if (m_timeOutCnt > 10000) //Time_out
	{
		aqprintf("*** TIMEOUT ***Completion failed! \n"); //just retry
		ChangePanelStatus(IMAGE_ACQUSITION_DONE);
		//m_timeOutCnt = 0;
		//vip_io_enable(HS_STANDBY); // Let's try without this (directly go to PANEL_ACTIVE)
		//Sleep(1000);
		//m_enPanelStatus = OPENNED; //move to first step..
		return false;
	}
	else
	{	
		if(m_pCrntStatus->qpi.Complete == TRUE)
		{			
			aqprintf("frames=%d complete=%d pulses=%d ready=%d\n",
				m_pCrntStatus->qpi.NumFrames,
				m_pCrntStatus->qpi.Complete,
				m_pCrntStatus->qpi.NumPulses,
				m_pCrntStatus->qpi.ReadyForPulse);	

			ChangePanelStatus(COMPLETE_SIGNAL_DETECTED);
			/*m_enPanelStatus = COMPLETE_SIGNAL_DETECTED;
			m_timeOutCnt = 0;
			aqprintf ("PSTAT5: COMPLETE_SIGNAL_DETECTED\n");*/

			//PC_GetImageHardware();
		}		
	}
	return true;
}

bool Acquire_4030e_child::PC_GetImageHardware()
{
	if (m_enPanelStatus != COMPLETE_SIGNAL_DETECTED)
	{
		aqprintf ("PC_GetImageHardware Error: panel status is not proper\n");
		return false;
	}

	int result = HCP_NO_ERR;
	QString errorStr;
	m_timeOutCnt += TIMEOUT_MAINLOOP;

	if(m_pCrntStatus->qpi.NumFrames >= m_iNumOfFramesRequested)			
	{	
		aqprintf ("before get_image_toBuf\n");

		result = vp->get_image_to_buf(m_pModeInfo->ColsPerFrame,m_pModeInfo->LinesPerFrame); //fill m_pCurrImage	

		aqprintf ("after get_image_toBuf\n");


		if (result == HCP_SAME_IMAGE_ERROR)
		{
			return true;
		}

		if (m_dlgControl->ChkAutoSendToDIPS)		
			vp->CopyFromBufAndSendToDips(dp);

		if (result != HCP_NO_ERR){
			errorStr = QString ("Error in sending image to currentImageBuffer. Error code is %1..Anyway, going to next step\n").arg(result);
			aqprintf(errorStr.toLocal8Bit().constData());
		}			
		else
		{
			//Display on the screen
			m_pCurrImage->FillPixMap(m_iCurWinMidVal, m_iCurWinWidthVal);
			m_pCurrImage->DrawToLabel(m_dlgControl->lbCurrentImage);

			//Save as raw file
			if (m_dlgControl->ChkAutoSave->isChecked())
			{
				QString folderPath = m_dlgControl->lineEditCurImageSaveFolder->text();				

				if (folderPath.length() < 2)
					folderPath = "C:";

				QString strFileName = QString("%1_CurImg").arg(this->idx); //panel number display
				QTime time = QTime::currentTime();    
				QString str = time.toString("_hh_mm_ss");        
				// = str; 
				strFileName.append(str);
				strFileName.prepend("\\");
				strFileName.prepend(folderPath);

				QString strGroupCommonName = strFileName;//before extension

				strFileName.append(".raw");
				m_pCurrImage->SaveDataAsRaw(strFileName.toStdString().c_str());

				if (m_dlgControl->ChkDarkCorrectedSave->isChecked())
				{
					if (!m_pCurrImageDarkCorrected->IsEmpty())
					{
						QString tmpDarkCorrImgPath = strGroupCommonName;
						tmpDarkCorrImgPath.append("_DARKCORR");
						tmpDarkCorrImgPath.append(".raw");
						m_pCurrImageDarkCorrected->SaveDataAsRaw(tmpDarkCorrImgPath.toStdString().c_str());
					}					
				}
				if (m_dlgControl->ChkRawSave->isChecked())
				{
					if (!m_pCurrImageRaw->IsEmpty())
					{
						QString tmpRawImgPath = strGroupCommonName;
						tmpRawImgPath.append("_RAW");
						tmpRawImgPath.append(".raw");
						m_pCurrImageRaw->SaveDataAsRaw(tmpRawImgPath.toStdString().c_str());
					}
				}

			}
		}
		

		ChangePanelStatus(IMAGE_ACQUSITION_DONE);
		//m_enPanelStatus = IMAGE_ACQUSITION_DONE; //even when wait_on_num_frames and sending image is failed..
		//m_timeOutCnt = 0;
		//aqprintf ("PSTAT6: IMAGE_ACQUSITION_DONE\n");
	}	
	return true;
}

bool Acquire_4030e_child::PC_CallForStanby() //also can be used for SW acquisition for reloop
{
	int result = vip_io_enable(HS_STANDBY);
	if (result != HCP_NO_ERR)
		aqprintf("Error on calling vip_io_enable. Result is %d\n", result);

	//Sleep(1000);
	ChangePanelStatus(STANDBY_CALLED);
	return true;
}


bool Acquire_4030e_child::PC_WaitForStanby() //also can be used for SW acquisition for reloop
{

	//any time it can be called for next purlpose

	/*if (m_enPanelStatus != IMAGE_ACQUSITION_DONE)
	{
		aqprintf ("PC_ReStandbyPanel Error: panel status is not proper\n");
		m_enPanelStatus = OPENNED;
		return false;
	}*/

	//m_pSysSemaphore->acquire();
	//vip_io_enable(HS_STANDBY);
	//m_pSysSemaphore->release(3);

	QString errorStr;
	int result = HCP_NO_ERR;
	//m_pCrntStatus->qpi.Complete = FALSE;	

	m_timeOutCnt += TIMEOUT_MAINLOOP;

	//m_pSysSemaphore->acquire();
	result = vp->query_prog_info (*m_pCrntStatus); // YK: Everytime, No data error 8
	//m_pSysSemaphore->release(3);

	if (result != HCP_NO_ERR) //Not that serious in this step
	{	
		errorStr = QString ("Error in WaitForStanby. Error code is %1. Now, retrying...\n").arg(result);		
		aqprintf(errorStr.toLocal8Bit().constData());
		//m_enPanelStatus = NOT_OPENNED;//Let's try with PANEL_ACTIVE (without reselection and HS_ACTIVE setting
		//Sleep(3000);
		//result = vip_io_enable(HS_ACTIVE);not work
		//aqprintf("result = %d", result);
		//vip_reset_state();
		//result = vip_reset_state(); not work
		//aqprintf(" vip_reset_state result = %d", result);

		ChangePanelStatus(IMAGE_ACQUSITION_DONE);
		//vip_io_enable(HS_STANDBY); // Let's try without this (directly go to PANEL_ACTIVE)
		return false;
	}
	else if (m_timeOutCnt > 5000) //Time_out  --> Impossible! should restart the panel
	{
		aqprintf("*** TIMEOUT ***Restanby failed! \n"); //just retry
		//vip_io_enable(HS_STANDBY);
		//m_timeOutCnt = 0;
		//m_enPanelStatus = IMAGE_ACQUSITION_DONE; //Re open the panel
		//aqprintf ("PSTAT6: IMAGE_ACQUSITION_DONE\n");
		ChangePanelStatus(IMAGE_ACQUSITION_DONE);
		return false;
	}
	else
	{	
		if(m_pCrntStatus->qpi.NumFrames == 0 && m_pCrntStatus->qpi.ReadyForPulse == 0) //0 0 0 0
		{			
			aqprintf("frames=%d complete=%d pulses=%d ready=%d\n",
				m_pCrntStatus->qpi.NumFrames,
				m_pCrntStatus->qpi.Complete,
				m_pCrntStatus->qpi.NumPulses,
				m_pCrntStatus->qpi.ReadyForPulse);			

			ChangePanelStatus(STANDBY_SIGNAL_DETECTED);

			/*m_enPanelStatus = STANDBY_SIGNAL_DETECTED;
			m_timeOutCnt = 0;
			aqprintf ("PSTAT7: STANBY_SIGNAL_DETECTED\n");			*/

			//m_TimerMainLoop->stop(); //will be resumed after getting message from parent //0404 edited -->to avoid "Hardware handshaking interference and image redundancy problem!
		}		
	}
	return true;
}


bool Acquire_4030e_child::PC_SoftwareAcquisition_SingleShot() //should be done at READY_FOR_PULSE
{
	if (m_enPanelStatus != READY_FOR_PULSE && m_enPanelStatus != STANDBY_SIGNAL_DETECTED)
	{
		aqprintf ("PC_SoftwareAcquisition_SingleShot Error: panel status is not proper\n");
		return false;
	}

	QMessageBox msgBox;
	msgBox.setText("Software handshaking image acquisition failed! Try once again.");	
	//msgBox.exec();	

	m_timeOutCnt = 0;
	m_dlgProgBar->setValue(0);
	//m_dlgProgBar->show();
	//m_dlgProgBar->setValue(0);
	m_dlgProgBar->show();

	if (!SWSingleAcquisition(*m_pCrntStatus))
	{
		//m_dlgProgBar->close();
		msgBox.exec();
	}	
	else
	{
		aqprintf("SWSingleAcquisition Success\n");
		//m_dlgProgBar->close();
	}

	//ChangePanelStatus(IMAGE_ACQUSITION_DONE);
	//ChangePanelStatus(OPENNED);


	//Sleep(300);
	return true;
}


bool Acquire_4030e_child::PC_DarkFieldAcquisition_SingleShot(int avgFrames) //should be done at READY_FOR_PULSE
{
	if (m_enPanelStatus != ACQUIRING_DARK_IMAGE)
	{
		aqprintf ("Dark field acquisition Error: panel status is not proper\n");
		return false;
	}
	//m_bAcquisitionOK = false;

	aqprintf("DARKFIELD_ACQUSITION_START\n");

	m_dlgProgBar->setValue(0);

	if (!PerformDarkFieldCalibration(*m_pCrntStatus, avgFrames))
	{
		//m_dlgProgBar->close();
		m_dlgProgBar->setValue(100);
		aqprintf("Error in dark field acquisition\n");		
	}
	else
	{
		//m_dlgProgBar->close();
		//aqprintf("Dark Field Acquisition Success\n");
		aqprintf("DARKFIELD_ACQUSITION_COMPLETE\n");
	}	
//
//	m_bAcquisitionOK = true;

	
	return true;
}


bool Acquire_4030e_child::LoadDarkImage(QString& filePath)
{
	QFileInfo fileInfo = QFileInfo(filePath);

	if (!fileInfo.exists())
		return false;

	//binary processing for gray 16 bit images	
	//aqprintf("Start LoadDark\n");
	int width = vp->m_iSizeX;
	int height = vp->m_iSizeY;

	if (width < 1 || height <1)
		return false;

	//if (!m_pDarkImage->IsEmpty())
	//	m_pDarkImage->ReleaseBuffer();

	if (!m_pDarkImage->LoadRawImage(filePath.toStdString().c_str(),width, height)) //Release Buffer is inside this func.
		aqprintf("Error on LoadRawImage\n");
	
	if (!m_pDarkImage->FillPixMap(DEFAULT_WINLEVEL_MID, DEFAULT_WINLEVEL_WIDTH))
		aqprintf("Error on FillPixMap\n");

	if (!m_pDarkImage->DrawToLabel(this->m_dlgControl->lbDarkField)) //SetPixMap 
		aqprintf("Error on drawing");

	return true;
}



bool Acquire_4030e_child::LoadGainImage(QString& filePath)
{
	QFileInfo fileInfo = QFileInfo(filePath);

	if (!fileInfo.exists())
		return false;

	//binary processing for gray 16 bit images	
	//aqprintf("Start LoadDark\n");
	int width = vp->m_iSizeX;
	int height = vp->m_iSizeY;

	if (width < 1 || height <1)
		return false;

	//if (!m_pDarkImage->IsEmpty())
	//	m_pDarkImage->ReleaseBuffer();

	if (!m_pGainImage->LoadRawImage(filePath.toStdString().c_str(),width, height)) //Release Buffer is inside this func.
		aqprintf("Error on Load Gain Image\n");

	if (!m_pGainImage->FillPixMap(DEFAULT_WINLEVEL_MID, DEFAULT_WINLEVEL_WIDTH))
		aqprintf("Error on FillPixMap\n");

	if (!m_pGainImage->DrawToLabel(this->m_dlgControl->lbGainField)) //SetPixMap 
		aqprintf("Error on drawing");

	return true;
}


void Acquire_4030e_child::ReDraw(int lowerWinVal, int upperWinVal)//current image only
{
	//only CurrentImg Redraw for time saving
	int midVal = (upperWinVal + lowerWinVal)/2.0;
	//if (midVal < 0 || midVal > 4095)
	if (midVal < 0 || midVal > 65535)
		midVal = DEFAULT_WINLEVEL_MID;

	int widthVal = upperWinVal - lowerWinVal;
	//if (widthVal < 0 || widthVal > 4095)
	if (widthVal < 0 || widthVal > 65535)
		widthVal = DEFAULT_WINLEVEL_WIDTH;


	m_iCurWinMidVal = midVal;
	m_iCurWinWidthVal = widthVal;	
	
	if (!m_pCurrImage->FillPixMap(midVal, widthVal))
		aqprintf("Error on FillPixMap\n");

	if (!m_pCurrImage->DrawToLabel(this->m_dlgControl->lbCurrentImage)) //SetPixMap 
		aqprintf("Error on drawing");

}



bool Acquire_4030e_child::GetGainImageFromCurrent()
{	
	if (m_pCurrImage->IsEmpty())
	{		
		return false;
	}

	m_pGainImage->CopyFromBuffer(m_pCurrImage->m_pData,
		m_pCurrImage->m_iWidth, m_pCurrImage->m_iHeight);


	int idx = this->m_iProcNum;
	//QString strFileName = QString("C:\\DarkImage_Avg%1").arg(SumSuccessCnt);
	QString strFolderName = m_OptionSettingChild.m_strGainImageSavingFolder[idx];

	double tmpMean = 0.0;
	double tmpSD = 0.0;
	double tmpMAX = 0.0;
	double tmpMIN = 0.0;

	m_pGainImage->CalcImageInfo(tmpMean, tmpSD, tmpMIN, tmpMAX);
	QString strMeanVal = QString("%1").arg((int)tmpMean);	

	QString strFileName = QString("\\%1_Gain").arg(idx);

	QDate date = QDate::currentDate();
	QTime time = QTime::currentTime();
	QString strDate = date.toString("_yyyy_MM_dd");
	QString strTime = time.toString("_hh_mm_ss_");
	strFileName.append(strDate);
	strFileName.append(strTime);
	
	strFileName.append(QString("M%1").arg(strMeanVal));
	strFileName.append(".raw");

	strFileName.prepend(strFolderName);
	
	if (!m_pGainImage->SaveDataAsRaw(strFileName.toLocal8Bit().constData()))
	{
		aqprintf("Cannot export to raw file\n");
	}

	//UpdateDarkImagePath
	m_dlgControl->lineEditGainPath->setText(strFileName);

	//if (!m_pDarkImage->FillPixMap(m_iCurWinMidVal, m_iCurWinWidthVal)) //16 bit to 8 bit
	//if (!m_pDarkImage->FillPixMap(DEFAULT_WINLEVEL_MID, DEFAULT_WINLEVEL_WIDTH)) //16 bit to 8 bit
	if (!m_pGainImage->FillPixMap(DEFAULT_WINLEVEL_MID, DEFAULT_WINLEVEL_WIDTH)) //0 - 3000 is enough
		aqprintf("Error on FillPixMap\n");

	if (!m_pGainImage->DrawToLabel(this->m_dlgControl->lbGainField)) //SetPixMap 
		aqprintf("Error on drawing");

	return true;	
}

void Acquire_4030e_child::ChangePanelStatus(PSTAT enStatus)
{
	m_bAcquisitionOK = true;

	m_timeOutCnt = 0;
	m_enPanelStatus = enStatus;

	switch (enStatus)
	{
	case NOT_OPENNED:
		aqprintf ("PSTAT0: NOT_OPENNED\n");	
		break;
	case OPENNED:
		aqprintf ("PSTAT1: OPENNED\n");
		break;
	case PANEL_ACTIVE:
		aqprintf ("PSTAT2: PANEL_ACTIVE\n");
		break;
	case READY_FOR_PULSE:
		aqprintf ("PSTAT3: READY_FOR_PULSE\n");		
		aqprintf("READY FOR X-RAYS - EXPOSE AT ANY TIME\n");		
		break;
	case PULSE_CHANGE_DETECTED:
		aqprintf ("PSTAT4: PULSE_CHANGE_DETECTED\n");		
		break;
	case COMPLETE_SIGNAL_DETECTED:
		aqprintf ("PSTAT5: COMPLETE_SIGNAL_DETECTED\n");		
		break;
	case IMAGE_ACQUSITION_DONE:
		aqprintf ("PSTAT6: IMAGE_ACQUSITION_DONE\n");		
		break;

	case STANDBY_CALLED:
		aqprintf ("PSTAT7: STANDBY_CALLED\n");		
		break;
	case STANDBY_SIGNAL_DETECTED:
		aqprintf ("PSTAT8: STANBY_SIGNAL_DETECTED\n");			
		break;
	case ACQUIRING_DARK_IMAGE:
		aqprintf ("PSTAT9: ACQUIRING_DARK_IMAGE\n");			
		break;
	}
}
