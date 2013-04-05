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
//#include "dlgprogbaryk.h"
#include <QProgressDialog>
#include <QSystemSemaphore>

#define TIMEOUT_MAINLOOP 50

Acquire_4030e_child::Acquire_4030e_child (int argc, char* argv[])
: QApplication (argc, argv)
{
	if (argc < 4) {
		aqprintf ("Error with commandline\n");
		exit (-1);
	}

	bool ok;
	this->idx = QString(argv[2]).toInt(&ok,10);
	if (!ok) {
		aqprintf ("Error with commandline\n");
		exit (-1);
	}

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


	m_TimerPollMsgFromParent = new QTimer(this);
	m_TimerMainLoop = new QTimer(this);

	connect(m_TimerPollMsgFromParent, SIGNAL(timeout()), this, SLOT(TimerPollMsgFromParent_event()));
	connect(m_TimerMainLoop, SIGNAL(timeout()), this, SLOT(TimerMainLoop_event()));


	m_dlgControl = new Acquire_4030e_DlgControl();
	//m_dlgProgBar = new DlgProgBarYK((QWidget*)m_dlgControl);
	m_dlgProgBar = new QProgressDialog("Operation in progress.", "Cancel", 0, 100);
	m_dlgProgBar->setWindowModality(Qt::WindowModal);
	//connect(m_dlgProgBar, SIGNAL(canceled()), this, SLOT(ProgressCanceled());

	m_timeOutCnt = 0;
	m_iNumOfFramesRequested = 1;

	m_pCrntStatus = NULL;
	m_pModeInfo = NULL;	

	m_iCurWinMidVal = DEFAULT_WINLEVEL_MID;
	m_iCurWinWidthVal= DEFAULT_WINLEVEL_WIDTH;

	
	
}

//void Acquire_4030e_child::m_timerPollMsg_event()
//{
//	aqprintf("Timer test\n");
//}

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

	//m_ImageInfoFout.close();

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
Acquire_4030e_child::init(const char* strProcNum, const char* strRecepterPath)
{  
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

	ReleaseMemory(); // for m_pDarkImage and pGain

	m_pCrntStatus = new UQueryProgInfo; //currentStatus    
	m_pModeInfo = new SModeInfo;	

	m_pDarkImage = new YK16GrayImage(); //offset image buffer
	m_pGainImage = new YK16GrayImage(); //Gain (flood or flat) image buffer
	m_pCurrImage = new YK16GrayImage();


	this->m_strProcNum = strProcNum;
	this->m_strReceptorPath = strRecepterPath;

	aqprintf ("Child %s got request to open panel: %s\n", strProcNum, strRecepterPath);

	this->idx = QString(strProcNum).toInt();   

	//if (dp != NULL)
	//	aqprintf("dp is not NULL\n");

	//Only do it once at first init.
	if (dp == NULL)
	{
		this->dp = new Dips_panel;
		dp->open_panel (this->idx, HIRES_IMAGE_HEIGHT, HIRES_IMAGE_WIDTH); //if failed, exit(1) -->very rare.    
		aqprintf ("DIPS shared memory created.\n");    
	}

	int result = open_receptor (strRecepterPath);	

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
		
	//this->m_pGainImage = new USHORT [vp->m_iSizeX * vp->m_iSizeY];

	this->m_bAcquisitionOK = true; 

	//YK: It is mandatory!!!
	vp->m_bDarkCorrApply = false;
	vp->m_bGainCorrApply = false;

	//m_TimerPollMsgFromParent->start(100);
	m_TimerPollMsgFromParent->start(50);
	m_TimerMainLoop->start(100);


	//QString strImgLogPath = QString("C:\\ImageInfo_panel_%1.txt").arg(m_strProcNum);	
	//m_ImageInfoFout.open(strImgLogPath.toStdString().c_str());	
	//m_ImageInfoFout << "TITLE" << "	" << "Mean" <<"	" << "SD"<< "	" << "MIN" <<"	"<<"MAX" << "	" << "ImageAcqusitionResult" << std::endl;
	
	//m_pSysSemaphore = new QSystemSemaphore("acquire_4030e",3, QSystemSemaphore::Create);
	
	
	return true;
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

int 
Acquire_4030e_child::open_receptor (const char* path)
{
	//aqprintf ("Log by YKP1_path= [%d] %s\n", idx,path);
	QMessageBox msgBox;
	
	//if there is no path exist, return false;    
	int result;
	//this->vp = new Varian_4030e (this->idx); //for each child vp will be assigned
	this->vp = new Varian_4030e (this->idx, this); //for each child vp will be assigned

	result = vp->open_link (idx, path); // idx of current child process

	if (result != HCP_NO_ERR)
		aqprintf("Error during open receptor. Error code = %d\n", result);
	
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
	m_enPanelStatus = COMPLETE_SIGNAL_DETECTED;  // go to standby

	this->m_bAcquisitionOK = true;
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
	result = vip_reset_state(); //Mandatory!!! or reopen the panel.

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

	m_dlgProgBar->setValue(10);	

	//Repeat below procedure //MultiFrame
	for (i = 0 ; i< avgFrameCnt ; i++)
	{		
		result = HCP_NO_ERR;
		m_timeOutCnt = 0;

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
					Sleep(500);
				}
			}
		}		

		//2] WAIT AC
		if (result == HCP_NO_ERR)
		{	    
			result = vp->query_prog_info (crntStatus);
			Sleep(100);

			if (result == HCP_NO_ERR)
			{	
				m_timeOutCnt = 0;
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
			//aqprintf("YK6\n");
			result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, FALSE);
			if (result != HCP_NO_ERR)
				aqprintf("vip_sw_handshaking(VIP_SW_VALID_XRAYS, FALSE) returns %d\n", result);
			result = vip_sw_handshaking(VIP_SW_PREPARE, FALSE);
			if (result != HCP_NO_ERR)
				aqprintf("vip_sw_handshaking(VIP_SW_PREPARE, FALSE) returns %d\n", result);
		}
		//Repeat above procedure //MultiFrame

		int tmpVal = (int)(80.0 / (double)(avgFrameCnt) * (i+1)); // 10 + 80 + 10
		m_dlgProgBar->setValue(tmpVal);
		aqprintf("Prog %d\n", tmpVal);
	} //end of for loop

	//in DualRadTest, this code is not existing
	result = vip_enable_sw_handshaking(FALSE);

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


	QString strFileName = QString("C:\\DarkImage_Avg%1").arg(SumSuccessCnt);
	QTime time = QTime::currentTime();    
	QString str = time.toString("_hh_mm_ss");        
	strFileName.append(str);
	strFileName.append(".raw");

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
			QTime time = QTime::currentTime();    
			QString str = time.toString("_hh_mm_ss");        
			strFileName.append(str);
			strFileName.append(".raw");

			strFileName.prepend("\\");
			strFileName.prepend(folderPath);
			m_pCurrImage->SaveDataAsRaw(strFileName.toStdString().c_str());	

		}
	}
	if (result == HCP_NO_ERR)
	{		
		result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, FALSE);
		if (result != HCP_NO_ERR)
			aqprintf("vip_sw_handshaking(VIP_SW_VALID_XRAYS, FALSE) returns %d\n", result);
		result = vip_sw_handshaking(VIP_SW_PREPARE, FALSE);
		if (result != HCP_NO_ERR)
			aqprintf("vip_sw_handshaking(VIP_SW_PREPARE, FALSE) returns %d\n", result);
	}

	m_dlgProgBar->setValue(100);

	//in DualRadTest, this code is not existing
	result = vip_enable_sw_handshaking(FALSE);
	
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
	aqprintf(m_strFromParent.toLocal8Bit().constData()); //PCOMMAND_..
	
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
		m_TimerPollMsgFromParent->stop();
		m_TimerMainLoop->stop(); //if running
		aqprintf("PRESPP_KILL\n");
	}	
	else if (m_strFromParent.contains("ACTIVATE"))//msg is sent from parent's main loop related to advantech
	{		
		aqprintf("PRESPP_ACTIVATE\n");
		
		//m_TimerMainLoop->start(100); //from standby (0 0 0 0) --> hardwarehandshaking & HS_ACTIVE
		m_bAcquisitionOK = true; //resume the main loop timer
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

void Acquire_4030e_child::TimerPollMsgFromParent_event()
{	
	QString tmpStr;
	char buffer[128];    
	memset (buffer,0, 128);        
	fgets(buffer,128,stdin);
	tmpStr = (const char*)buffer;

	fflush(stdin);	

	if (tmpStr.length() > 1)
	{		
		m_strFromParent=tmpStr;		
		aqprintf(m_strFromParent.toLocal8Bit().constData());
		InterpretAndFollow ();
	}	
	return;	
}

void Acquire_4030e_child::TimerMainLoop_event() //called every 50 ms
{
	if (!m_bAcquisitionOK)
		return;
	//stop main loop to go into the sub loop
	//At the end of the sub-loop code, resume the main loop	
	int result = HCP_NO_ERR;
	QString errorStr;

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
		m_bAcquisitionOK = false; //BUSY flag
		PC_WaitForPulse();
		m_bAcquisitionOK = true; //BUSY flag
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

	case IMAGE_ACQUSITION_DONE:
		m_bAcquisitionOK = false; //BUSY flag
		PC_WaitForStanby(); //inside here image acquisition
		m_bAcquisitionOK = true; //BUSY flag
		break;            	    	

	case STANDBY_SIGNAL_DETECTED:
		m_bAcquisitionOK = false; //Just hold the timer
		m_enPanelStatus = OPENNED;

		//stop here the loop until getting ACTIVATION msg from parent

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
	if (!init(this->m_strProcNum.toLocal8Bit().constData(), this->m_strReceptorPath.toLocal8Bit().constData()))
	{			
		aqprintf ("Cannot Open the panel! Loop will be stopped.\n");
		m_bAcquisitionOK = false;		
	}
	else
		m_enPanelStatus = OPENNED;	

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
	aqprintf ("PSTAT1: OPENNED\n");

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
			Sleep(300);
		}
		return false;
	}				
		
	result = vip_io_enable (HS_ACTIVE); // Frame | Complete | NumPulse | PanelReady		

	if (result != HCP_NO_ERR) {	
		aqprintf("**** returns error %d - Panel Activation Error\n", result);		
	}
	m_enPanelStatus = PANEL_ACTIVE;
	aqprintf ("PSTAT2: PANEL_ACTIVE\n");	

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
		vip_io_enable(HS_STANDBY); //should be!!
		m_enPanelStatus = OPENNED; //Go back to first step		
	}
	else if (m_timeOutCnt > 5000) //Time_out //5000 is fixed! Optimized value
	{
		aqprintf("*** TIMEOUT ***_wait_on_ready_for_pulse\n"); //just retry
		
		aqprintf("frames=%d complete=%d pulses=%d ready=%d\n",
				m_pCrntStatus->qpi.NumFrames,
				m_pCrntStatus->qpi.Complete,
				m_pCrntStatus->qpi.NumPulses,
				m_pCrntStatus->qpi.ReadyForPulse);

		m_timeOutCnt = 0;
		vip_io_enable(HS_STANDBY);
		Sleep(1000);
		m_enPanelStatus = OPENNED;
		return false;
	}
	else
	{
		if (m_pCrntStatus->qpi.ReadyForPulse == 1)
		{			
			m_enPanelStatus = READY_FOR_PULSE;			
			aqprintf ("PSTAT3: READY_FOR_PULSE\n"); 
			m_timeOutCnt = 0;		
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
		m_enPanelStatus = OPENNED;//go back to first step	
		vip_io_enable(HS_STANDBY);
		return false;
	}
	else if (m_timeOutCnt > 2000) 
	{		
		m_enPanelStatus = IMAGE_ACQUSITION_DONE; //GOTO Standby
		return false;
	}
	else
	{
		if (m_pCrntStatus->qpi.NumPulses != prevNumPulses) //if thiere is any change		
		{
			m_enPanelStatus = PULSE_CHANGE_DETECTED;
			aqprintf ("PSTAT4: PULSE_CHANGE_DETECTED\n");
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

		vip_io_enable(HS_STANDBY); // Let's try without this (directly go to PANEL_ACTIVE)
		Sleep(1000);
		m_enPanelStatus = OPENNED;//Let's try with PANEL_ACTIVE (without reselection and HS_ACTIVE setting		
		return false;
	}
	else if (m_timeOutCnt > 10000) //Time_out
	{
		aqprintf("*** TIMEOUT ***Completion failed! \n"); //just retry
		m_timeOutCnt = 0;
		vip_io_enable(HS_STANDBY); // Let's try without this (directly go to PANEL_ACTIVE)
		Sleep(1000);
		m_enPanelStatus = OPENNED; //move to first step..
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

			m_enPanelStatus = COMPLETE_SIGNAL_DETECTED;
			m_timeOutCnt = 0;
			aqprintf ("PSTAT5: COMPLETE_SIGNAL_DETECTED\n");

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
		result = vp->get_image_to_buf(m_pModeInfo->ColsPerFrame,m_pModeInfo->LinesPerFrame); //fill m_pCurrImage	

		if (result == HCP_SAME_IMAGE_ERROR)
		{
			return true;
		}

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
				strFileName.append(str);
				strFileName.append(".raw");

				strFileName.prepend("\\");
				strFileName.prepend(folderPath);
				m_pCurrImage->SaveDataAsRaw(strFileName.toStdString().c_str());			
			}
		}
		
		m_enPanelStatus = IMAGE_ACQUSITION_DONE; //even when wait_on_num_frames and sending image is failed..

		m_timeOutCnt = 0;
		aqprintf ("PSTAT6: IMAGE_ACQUSITION_DONE\n");
	}	
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
	vip_io_enable(HS_STANDBY);
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
		m_enPanelStatus = NOT_OPENNED;//Let's try with PANEL_ACTIVE (without reselection and HS_ACTIVE setting
		//vip_io_enable(HS_STANDBY); // Let's try without this (directly go to PANEL_ACTIVE)
		return false;
	}
	else if (m_timeOutCnt > 5000) //Time_out  --> Impossible! should restart the panel
	{
		aqprintf("*** TIMEOUT ***Restanby failed! \n"); //just retry
		m_timeOutCnt = 0;
		m_enPanelStatus = OPENNED; //Re open the panel
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

			m_enPanelStatus = STANDBY_SIGNAL_DETECTED;
			m_timeOutCnt = 0;
			aqprintf ("PSTAT7: STANBY_SIGNAL_DETECTED\n");	
			aqprintf("READY FOR X-RAYS - EXPOSE AT ANY TIME\n");

			//m_TimerMainLoop->stop(); //will be resumed after getting message from parent //0404 edited -->to avoid "Hardware handshaking interference and image redundancy problem!
		}		
	}
	return true;
}


bool Acquire_4030e_child::PC_SoftwareAcquisition_SingleShot() //should be done at READY_FOR_PULSE
{
	if (m_enPanelStatus != READY_FOR_PULSE)
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

	Sleep(300);
	return true;
}


bool Acquire_4030e_child::PC_DarkFieldAcquisition_SingleShot(int avgFrames) //should be done at READY_FOR_PULSE
{
	if (m_enPanelStatus != READY_FOR_PULSE)
	{
		aqprintf ("PC_SoftwareAcquisition_SingleShot Error: panel status is not proper\n");
		return false;
	}	

	m_dlgProgBar->setValue(0);
	//m_dlgProgBar->show();

	if (!PerformDarkFieldCalibration(*m_pCrntStatus, avgFrames))
	{
		//m_dlgProgBar->close();
		m_dlgProgBar->setValue(100);
		aqprintf("Error in dark field acquisition\n");		
	}
	else
	{
		//m_dlgProgBar->close();
		aqprintf("Dark Field Acquisition Success\n");
	}

	Sleep(300);		

	//if current is hardware handshaking:

	if (m_dlgControl->RadioHardHandshakingEnable->isChecked())
		m_enPanelStatus = OPENNED;


	return true;
}


bool Acquire_4030e_child::LoadDarkImage(QString& filePath)
{
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

	QString strFileName = QString("C:\\GainImage");
	QTime time = QTime::currentTime();    
	QString str = time.toString("_hh_mm_ss");        
	strFileName.append(str);
	strFileName.append(".raw");
	
	if (!m_pGainImage->SaveDataAsRaw(strFileName.toLocal8Bit().constData()))
	{
		aqprintf("Cannot export to raw file\n");
	}

	//UpdateDarkImagePath
	m_dlgControl->lineEditGainPath->setText(strFileName);

	//if (!m_pDarkImage->FillPixMap(m_iCurWinMidVal, m_iCurWinWidthVal)) //16 bit to 8 bit
	//if (!m_pDarkImage->FillPixMap(DEFAULT_WINLEVEL_MID, DEFAULT_WINLEVEL_WIDTH)) //16 bit to 8 bit
	if (!m_pDarkImage->FillPixMap(DEFAULT_WINLEVEL_MID, DEFAULT_WINLEVEL_WIDTH)) //0 - 3000 is enough
		aqprintf("Error on FillPixMap\n");

	if (!m_pDarkImage->DrawToLabel(this->m_dlgControl->lbGainField)) //SetPixMap 
		aqprintf("Error on drawing");

	return true;	
}