#include "Acquire_4030e_DlgControl.h"
#include "Acquire_4030e_child.h"
#include <QCheckBox>
#include <QFileDialog>
#include <QMessageBox>
#include "aqprintf.h"
#include "varian_4030e.h"

Acquire_4030e_DlgControl::Acquire_4030e_DlgControl(): QDialog ()
{
    /* Sets up the GUI */
    setupUi (this);

	this->SpinDarkAvgFrames->setValue(4);

	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;

	
	QString str= QString("%1").arg(qMyApp->idx);
	EditPanelIndex->setText(str);
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
	
	if (ChkDarkFieldApply->isChecked())    
    {
		/*((Acquire_4030e_child*)qApp)->StartCommandTimer(m_iPanelIdx,
		Acquire_4030e_parent::DARK_CORR_APPLY_ON);	*/
		qMyApp->vp->m_bDarkCorrApply = true;		
    }
    else
    {
		qMyApp->vp->m_bDarkCorrApply = false;	
    }
}
void Acquire_4030e_DlgControl::ChkGainCorrectionOn()
{
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;	

	if(ChkGainCorrectionApply->isChecked())
	{
		qMyApp->vp->m_bGainCorrApply = true;
	}
	else
	{
		qMyApp->vp->m_bGainCorrApply = false;
	}
}


void Acquire_4030e_DlgControl::closeEvent(QCloseEvent *event)
{
	hide();	
	event->ignore();
}


void Acquire_4030e_DlgControl::OpenDarkImage() //Btn
{
	Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;

	QString filePath = QFileDialog::getOpenFileName(this, "Open Image","", "Image Files (*.raw)",0,0);

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

	QString filePath = QFileDialog::getOpenFileName(this, "Open Image","", "Image Files (*.raw)",0,0);

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
	//Acquire_4030e_child* qMyApp = (Acquire_4030e_child*)qApp;
	QDir dir = QDir::current(); //changeable?
	QString startFolder = dir.path();

	QString strFolderPath = QFileDialog::getExistingDirectory(this, "Open Directory",startFolder,
		QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

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