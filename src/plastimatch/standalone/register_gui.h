#ifndef REGISTER_GUI_H
#define REGISTER_GUI_H

#include <QtGui/QMainWindow>
#include "ui_register_gui.h"
#include <QStringList>
#include <vector>
//#include <fstream>

using namespace std;

class Gamma_parms;

class register_gui : public QMainWindow
{
    Q_OBJECT

public:
    register_gui(QWidget *parent = 0, Qt::WFlags flags = 0);
    ~register_gui();
 //   QString CorrectSingle_NKI2MHA(const char* filePath);
 //   QString CorrectSingle_NKI2DCM(const char* filePath);
	//QString CorrectSingle_NKI2DCM( const char* filePath, QString patientID, QString patientName);

 //   QString CorrectSingle_NKI2RAW( const char* filePath );

 //   QString CorrectSingle_MHA2DCM(const char* filePath );
	//QString CorrectSingle_MHA2DCM( const char* filePath, QString patientID, QString patientName);
    
    QString GammaMain(Gamma_parms* parms);
 

    public slots:
        //void SLT_OpenOffsetFile();
        //void SLT_OpenGainFile();
        //void SLT_OpenBadpixelFile();
        void SLT_Load_RD_Ref();
        void SLT_Load_RD_Comp();
        void SLT_RunBatchGamma();
        //void SLT_Correct_NKI2MHA(); //NKI to MHA
        //void SLT_Correct_NKI2DCM(); //NKI to MHA
        //void SLT_Correct_NKI2RAW(); //NKI to RAW: signed short!

public:
    //YK16GrayImage* m_pImgOffset;
    //YK16GrayImage* m_pImgGain;
    //Badpixmap;

    //vector<BADPIXELMAP> m_vPixelReplMap;	
    //vector<YK16GrayImage*> m_vpRawImg;
    QStringList m_strlistPath_RD_Ref;
    QStringList m_strlistPath_RD_Comp;

    QStringList m_strlistFileBaseName_Ref;
    QStringList m_strlistFileBaseName_Comp;
    
    QStringList m_strlistBatchReport;
    
    QStringList m_strlistPath_Output_Gammamap;
    QStringList m_strlistPath_Output_Failure;
    QStringList m_strlistPath_Output_Report;


private:
    Ui::register_guiClass ui;
};

#endif // gamma_gui_H
