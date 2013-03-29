#pragma once
#include <QDialog>
#include "ui_acquire_4030e_panel.h"
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

public:
    Acquire_4030e_DlgControl();
    ~Acquire_4030e_DlgControl();

public:
    int m_iPanelIdx;
    //bool m_bOffsetCorrOn; //offset corr = dark field corr
    //bool m_bGainCorrOn; //offset corr = dark field corr
	void closeEvent(QCloseEvent *event);	
	//bool DlgLoadDarkImage(QString& filePath);
	
};
