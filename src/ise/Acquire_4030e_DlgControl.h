#pragma once
#include <QDialog>
#include "ui_acquire_4030e_panel.h"
#include "acquire_4030e_define.h"
//#include <QMainWindow>
//#include "ui_acquire_4030e_window.h"
class QString;

class Acquire_4030e_DlgControl : public QDialog,
    public Ui::DlgPanelControl
{
    Q_OBJECT
    ;

public slots:
    void CloseLink();
    void OpenLink();
    void Run();
    void Pause();
    void EndProc();
    void RestartProc();    
    void HardHandshakingOn();    
    void SoftHandshakingOn();
    void SoftBeamOn();    
    void GetDarkFieldImage();
    void ChkOffsetCorrectionOn();
    void ChkGainCorrectionOn();
    void GetPanelInfo();    
	void OpenDarkImage();
	void OpenGainImage(); //Btn
	void OpenCurImgFolder();
	void ReDrawImg();
	void CopyCurrentImageForGainImage();
	void SaveSettingAsDefault_Child();
	void UpdateGUIFromSetting_Child();
	void SingleGainOn();
	void MultiGainOn();
	void ReLoadCalibImages();

public:
    Acquire_4030e_DlgControl();
    ~Acquire_4030e_DlgControl();

public:
    int m_iPanelIdx;
    //bool m_bOffsetCorrOn; //offset corr = dark field corr
    //bool m_bGainCorrOn; //offset corr = dark field corr
	void closeEvent(QCloseEvent *event);	
	//bool DlgLoadDarkImage(QString& filePath);
	

	
	//bool SaveSettingAsDefault_Child();
	//bool UpdateGUIFromSetting_Child();// it can be used as go_back to default setting
};
