#include "Acquire_4030e_DlgControl.h"
#include "Acquire_4030e_child.h"
#include <QCheckBox>
#include <QFileDialog>
#include <QMessageBox>
#include "aqprintf.h"
#include "varian_4030e.h"
//#include "YKOptionSetting.h"

Acquire_4030e_DlgControl::Acquire_4030e_DlgControl(): QDialog ()
{
    /* Sets up the GUI */
    setupUi (this);

	this->SpinDarkAvgFrames->setValue(4);

	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;

	
	QString str= QString("%1").arg(qMyApp->idx);
	//EditPanelIndex->setText(str);
}

Acquire_4030e_DlgControl::~Acquire_4030e_DlgControl()
{
}

void Acquire_4030e_DlgControl::CloseLink()
{
    //((Acquire_4030e_parent*)qApp)->SendCommandToChild(m_iPanelIdx,
//	Acquire_4030e_parent::CLOSE_PANEL);
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;

	if (!qMyApp->close_receptor())
	{
		aqprintf("PCMSG_CLOSING_RECEPTOR_ERROR\n");		
	}
	else
	{
		qMyApp->m_enPanelStatus = NOT_OPENNED;
		qMyApp->m_bAcquisitionOK = false; //Timer is still on but dummy loop
		aqprintf("PCMSG_RECEPTOR_CLOSED\n");
		//or timer off
	}
	return;
}
void Acquire_4030e_DlgControl::OpenLink()
{
    //((Acquire_4030e_child*)qApp)->StartCommandTimer(m_iPanelIdx,
	//Acquire_4030e_child::OPEN_PANEL);
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	qMyApp->m_enPanelStatus = NOT_OPENNED;
	qMyApp->m_bAcquisitionOK = true; //or Timer start
	aqprintf("PCMSG_RECEPTOR_OPENNED\n");
	
	return;
}
void Acquire_4030e_DlgControl::Run()
{    
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	qMyApp->m_bAcquisitionOK = true; 
	aqprintf("PCMSG_LOOP_RESUMED\n");
}
void Acquire_4030e_DlgControl::Pause()
{
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	qMyApp->m_bAcquisitionOK = false; 
	aqprintf("PCMSG_LOOP_STOPPED\n");    
}
void Acquire_4030e_DlgControl::EndProc() //KILL
{
    //((Acquire_4030e_child*)qApp)->StartCommandTimer(m_iPanelIdx,
	//Acquire_4030e_parent::KILL);
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	CloseLink();
	qMyApp->quit();
}
void Acquire_4030e_DlgControl::RestartProc()
{
	CloseLink();
	aqprintf("PRESPP_RESTART\n");//parent will do it    
}
void Acquire_4030e_DlgControl::HardHandshakingOn()
{
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;

	/*if (RadioHardHandshakingEnable->isChecked())
	{
		aqprintf("Already checked\n");
		return;
	}*/
	if (qMyApp->m_enPanelStatus != READY_FOR_PULSE) //should be done in standby status
	{
		aqprintf("PCMSG_pannel is not ready\n");
		RadioHardHandshakingEnable->setChecked(false);
		RadioSoftHandshakingEnable->setChecked(true);		
		return;
	}	
	//qMyApp->m_bSoftHandshakingEnable = false;

	qMyApp->m_bAcquisitionOK = true;
	qMyApp->m_enPanelStatus = OPENNED; //re loop
	
	aqprintf("PCMSG_HARDHANDSHAKING_ON\n");
}



void Acquire_4030e_DlgControl::SoftHandshakingOn()
{	
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;

	/*if (RadioSoftHandshakingEnable->isChecked())
	{
		aqprintf("Already checked\n");
		return;
	}*/

	if (qMyApp->m_enPanelStatus != READY_FOR_PULSE) //should be done in standby status
	{
		aqprintf("Pannel is not ready\n");		
		RadioHardHandshakingEnable->setChecked(true);
		RadioSoftHandshakingEnable->setChecked(false);
		return;
	}
	
	//qMyApp->m_bSoftHandshakingEnable = true;
	qMyApp->m_bAcquisitionOK = false; //stop the main loop
	//qMyApp->m_bAcquisitionOK = false;	
	//qMyApp->m_enPanelStatus = OPENNED;
	aqprintf("PCMSG_HARDHANDSHAKING_ON\n");
}

void Acquire_4030e_DlgControl::SoftBeamOn()
{    
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;

	if (qMyApp->m_dlgControl->RadioSoftHandshakingEnable->isChecked())
	{
		aqprintf("PCMSG_SOFTHANDSHAKING_ACQUSITION_START\n");
		qMyApp->PC_SoftwareAcquisition_SingleShot();
		aqprintf("PCMSG_SOFTHANDSHAKING_ACQUSITION_COMPLETE\n");
	}
	else
	{
		aqprintf ("First, change the handshaking status\n");
	}		
}

void Acquire_4030e_DlgControl::GetPanelInfo()
{
    //((Acquire_4030e_child*)qApp)->StartCommandTimer(m_iPanelIdx,
	//Acquire_4030e_parent::GET_PANEL_INFO);    

	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	//Go one more stanby cycle
	qMyApp->m_enPanelStatus = COMPLETE_SIGNAL_DETECTED; //Standby once again
}

void Acquire_4030e_DlgControl::GetDarkFieldImage()
{  
	int iAvgFrames = SpinDarkAvgFrames->value();

	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	aqprintf("PCMSG_DARKFIELD_ACQUSITION_START\n");
	qMyApp->PC_DarkFieldAcquisition_SingleShot(iAvgFrames);
	aqprintf("PCMSG_DARKFIELD_ACQUSITION_COMPLETE\n");
}

void Acquire_4030e_DlgControl::ChkOffsetCorrectionOn()
{  
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;	
	
	//if (ChkDarkFieldApply->isChecked())    
 //   {
	//	/*((Acquire_4030e_child*)qApp)->StartCommandTimer(m_iPanelIdx,
	//	Acquire_4030e_parent::DARK_CORR_APPLY_ON);	*/
	//	qMyApp->vp->m_bDarkCorrApply = true;		
 //   }
 //   else
 //   {
	//	qMyApp->vp->m_bDarkCorrApply = false;	
 //   }
}
void Acquire_4030e_DlgControl::ChkGainCorrectionOn()
{
	/*Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;	

	if(ChkGainCorrectionApply->isChecked())
	{
		qMyApp->vp->m_bGainCorrApply = true;
	}
	else
	{
		qMyApp->vp->m_bGainCorrApply = false;
	}*/
}


void Acquire_4030e_DlgControl::closeEvent(QCloseEvent *event)
{
	hide();	
	event->ignore();
}


void Acquire_4030e_DlgControl::OpenDarkImage() //Btn
{
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;

	int idx =qMyApp->m_iProcNum;

	QString defaultSearchFolder = qMyApp->m_OptionSettingChild.m_strDarkImageSavingFolder[idx];

	aqprintf("defaultSerchFolderForDark. ProcNum = %d, =  %s\n",idx,defaultSearchFolder.toLocal8Bit().constData());

	QString filePath;
	if (defaultSearchFolder.length() < 2)
	{
		
		filePath = QFileDialog::getOpenFileName(this, "Open Image","", "Image Files (*.raw)",0,0);
	}
	else
		filePath = QFileDialog::getOpenFileName(this, "Open Image",defaultSearchFolder, "Image Files (*.raw)",0,0);

	if (filePath.length() < 3)
		return;

	lineEditDarkPath->setText(filePath);
	//Fill Dark Image Array with this file

	QMessageBox dlgMsgBox;
	
	if (!qMyApp->LoadDarkImage(filePath))
	{
		dlgMsgBox.setText("Error on loading dark image!");
		dlgMsgBox.exec();
	}		
}


void Acquire_4030e_DlgControl::OpenGainImage() //Btn
{
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;

	//QString filePath;
	int idx =qMyApp->m_iProcNum;
	
	//= QFileDialog::getOpenFileName(this, "Open Image","", "Image Files (*.raw)",0,0);

	QString defaultSearchFolder = qMyApp->m_OptionSettingChild.m_strGainImageSavingFolder[idx];

	aqprintf("defaultSerchFolderForGain =  %s\n",defaultSearchFolder.toLocal8Bit().constData());

	QString filePath;
	if (defaultSearchFolder.length() < 2)
	{
		filePath = QFileDialog::getOpenFileName(this, "Open Gain Image","", "Image Files (*.raw)",0,0);
	}
	else
	{
		filePath = QFileDialog::getOpenFileName(this, "Open Gain Image",defaultSearchFolder, "Image Files (*.raw)",0,0);
	}

	if (filePath.length() < 3)
		return;


	lineEditGainPath->setText(filePath);
	//Fill Dark Image Array with this file

	QMessageBox dlgMsgBox;

	if (!qMyApp->LoadGainImage(filePath))
	{
		dlgMsgBox.setText("Error on loading gain image!");
		dlgMsgBox.exec();
	}		
}


void Acquire_4030e_DlgControl::OpenCurImgFolder()
{
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	
	
	int idx = qMyApp->m_iProcNum;
	QString startFolder = qMyApp->m_OptionSettingChild.m_strAcqFileSavingFolder[idx];
	if (startFolder.length() < 3)
	{
		QDir dir = QDir::current(); //changeable?
		startFolder = dir.path();
	}	

	QString strFolderPath = QFileDialog::getExistingDirectory(this, "Open Directory",startFolder,
		QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

	if (strFolderPath.length() < 2)
		return;


	this->lineEditCurImageSaveFolder->setText(strFolderPath);
}

void Acquire_4030e_DlgControl::ReDrawImg()
{
	int lowerVal = spinLowerVal->value();
	int upperVal = spinUpperVal->value();

	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	qMyApp->ReDraw(lowerVal, upperVal);//current image only
}

void Acquire_4030e_DlgControl::CopyCurrentImageForGainImage()
{
	QMessageBox msgBox;	
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	/*if (qMyApp->m_pCurrImage == NULL) // created during init.
		return;*/

	if (!qMyApp->GetGainImageFromCurrent())
	{
		aqprintf("Error on Getting Gain Image from current image. \n");
		msgBox.setText("Error on Getting Gain Image from current image!");
		msgBox.exec();
		return;
	}	
}


//Save this setting as a default setting when used during software starts.
void Acquire_4030e_DlgControl::SaveSettingAsDefault_Child()
{
	//1. Update setting data
	//2. Export as a file
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	YKOptionSetting* pOptionSetting = &(((Acquire_4030e_child*)qApp)->m_OptionSettingChild);
	//pOptionSetting->m_strPrimaryLogPath;
	int idx = qMyApp->m_iProcNum;
	

	//Image Panel directory should be set	
	pOptionSetting->m_strDriverFolder[idx] = this->EditDriverPath->text();

	QDir tmpDir = QDir(pOptionSetting->m_strDriverFolder[idx]);
	if (!tmpDir.exists())
	{
		aqprintf("Cannot save this setting.Panel driver should be found!\n");
		return;
	}	

	pOptionSetting->m_iWinLevelUpper[idx] = this->spinUpperVal->value();
	pOptionSetting->m_iWinLevelLower[idx] = this->spinLowerVal->value();;	

	/* File Save */
 	pOptionSetting->m_bSaveToFileAfterAcq[idx] = this->ChkAutoSave->isChecked();
	pOptionSetting->m_bSendImageToDipsAfterAcq[idx] = this->ChkAutoSendToDIPS->isChecked();
	pOptionSetting->m_bSaveDarkCorrectedImage[idx] = this->ChkDarkCorrectedSave->isChecked();
	pOptionSetting->m_bSaveRawImage[idx] = this->ChkRawSave->isChecked();
	pOptionSetting->m_strAcqFileSavingFolder[idx] = this->lineEditCurImageSaveFolder->text();

	/* Panel Control */
	pOptionSetting->m_bSoftwareHandshakingEnabled[idx] = this->RadioSoftHandshakingEnable->isChecked(); //false = hardware handshaking

	/*Dark Image Correction */
	pOptionSetting->m_bDarkCorrectionOn[idx] = this->ChkDarkFieldApply->isChecked();
	pOptionSetting->m_iDarkFrameNum[idx] =  this->SpinDarkAvgFrames->value();
	//m_strDarkImageSavingFolder[idx] = this->;
	pOptionSetting->m_strDarkImagePath[idx] = this->lineEditDarkPath->text(); //should be set later
	pOptionSetting->m_bTimerAcquisitionEnabled[idx] = this->ChkAutoDarkOn->isChecked(); //not implemented yet
	pOptionSetting->m_iTimerAcquisitionMinInterval[idx] = this->SpinTimeInterval->value();//not implemented yet

	//YK TEMP: not implemented yet. leave it as default value
	/*m_fDarkCufoffUpperMean[idx] = ;
	m_fDarkCufoffLowerMean[idx] = ;

	m_fDarkCufoffUpperSD[idx] = ;
	m_fDarkCufoffLowerSD[idx] = ;	*/

	/*Gain Image Correction */
	pOptionSetting->m_bGainCorrectionOn[idx] = this->ChkGainCorrectionApply->isChecked();
	pOptionSetting->m_bMultiLevelGainEnabled[idx] = this->RadioMultiGain->isChecked(); // false = single gain correction	
	//m_strGainImageSavingFolder[idx] = leave it as default
	pOptionSetting->m_strSingleGainPath[idx] = this->lineEditGainPath->text();
	pOptionSetting->m_fSingleGainCalibFactor[idx] = (this->lineEditSingleCalibFactor->text()).toDouble();
	


	pOptionSetting->ExportChildOption(pOptionSetting->m_defaultChildOptionPath[idx], idx);

}


//should be called when start
void Acquire_4030e_DlgControl::UpdateGUIFromSetting_Child() // it can be used as go_back to default setting
{
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	YKOptionSetting* pOptionSetting = &(((Acquire_4030e_child*)qApp)->m_OptionSettingChild);
	int idx = qMyApp->m_iProcNum;

	EditDriverPath->setText(pOptionSetting->m_strDriverFolder[idx]);
			
	spinUpperVal->setValue(pOptionSetting->m_iWinLevelUpper[idx]);
	spinLowerVal->setValue(pOptionSetting->m_iWinLevelLower[idx]);	

	///* File Save */
	ChkAutoSave->setChecked(pOptionSetting->m_bSaveToFileAfterAcq[idx]);
	ChkAutoSendToDIPS->setChecked(pOptionSetting->m_bSendImageToDipsAfterAcq[idx]);
	ChkDarkCorrectedSave->setChecked(pOptionSetting->m_bSaveDarkCorrectedImage[idx]);
	ChkRawSave->setChecked(pOptionSetting->m_bSaveRawImage[idx]);		
	lineEditCurImageSaveFolder->setText(pOptionSetting->m_strAcqFileSavingFolder[idx]);
	

	RadioSoftHandshakingEnable->setChecked(pOptionSetting->m_bSoftwareHandshakingEnabled[idx]);	
	RadioHardHandshakingEnable->setChecked(!pOptionSetting->m_bSoftwareHandshakingEnabled[idx]);	
	

	this->ChkDarkFieldApply->setChecked(pOptionSetting->m_bDarkCorrectionOn[idx]);

	///*Dark Image Correction */
	//pOptionSetting->m_bDarkCorrectionOn[idx] = this->ChkDarkFieldApply->isChecked();

	//pOptionSetting->m_iDarkFrameNum[idx] =  this->SpinDarkAvgFrames->value();
	this->SpinDarkAvgFrames->setValue(pOptionSetting->m_iDarkFrameNum[idx]);

	////m_strDarkImageSavingFolder[idx] = 

	//pOptionSetting->m_strDarkImagePath[idx] = this->lineEditDarkPath->text(); //should be set later
	lineEditDarkPath->setText(pOptionSetting->m_strDarkImagePath[idx]);

	//aqprintf(pOptionSetting->m_strDarkImagePath[idx].toLocal8Bit().constData());


	this->ChkAutoDarkOn->setChecked(pOptionSetting->m_bTimerAcquisitionEnabled[idx]);
	this->SpinTimeInterval->setValue(pOptionSetting->m_iTimerAcquisitionMinInterval[idx]);
	
	////YK TEMP: not implemented yet. leave it as default value
	///*m_fDarkCufoffUpperMean[idx] = ;
	//m_fDarkCufoffLowerMean[idx] = ;

	//m_fDarkCufoffUpperSD[idx] = ;
	//m_fDarkCufoffLowerSD[idx] = ;	*/

	///*Gain Image Correction */
	//pOptionSetting->m_bGainCorrectionOn[idx] = this->ChkGainCorrectionApply->isChecked();
	this->ChkGainCorrectionApply->setChecked(pOptionSetting->m_bGainCorrectionOn[idx]);


	//pOptionSetting->m_bMultiLevelGainEnabled[idx] = this->RadioMultiGain->isChecked(); // false = single gain correction	
	this->RadioMultiGain->setChecked(pOptionSetting->m_bMultiLevelGainEnabled[idx]);
	this->RadioSingleGain->setChecked(!pOptionSetting->m_bMultiLevelGainEnabled[idx]);

	////m_strGainImageSavingFolder[idx] = leave it as default
	//pOptionSetting->m_strSingleGainPath[idx] = this->lineEditGainPath->text();
	this->lineEditGainPath->setText(pOptionSetting->m_strSingleGainPath[idx]);

	//pOptionSetting->m_fSingleGainCalibFactor[idx] = (this->lineEditSingleCalibFactor->text()).toDouble();	
	QString tmpStr = QString("%1").arg(pOptionSetting->m_fSingleGainCalibFactor[idx]);
	this->lineEditSingleCalibFactor->setText(tmpStr);	

	ReLoadCalibImages();
}

void Acquire_4030e_DlgControl::SingleGainOn()
{
	this->RadioSingleGain->setChecked(true);
	this->RadioMultiGain->setChecked(false);
}
void Acquire_4030e_DlgControl::MultiGainOn()
{
	this->RadioSingleGain->setChecked(false);
	this->RadioMultiGain->setChecked(true);
}

void Acquire_4030e_DlgControl::ReLoadCalibImages()
{
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	qMyApp->LoadDarkImage(this->lineEditDarkPath->text());
	qMyApp->LoadGainImage(this->lineEditGainPath->text());
}
