#ifndef GAMMA_GUI_H
#define GAMMA_GUI_H

#include <QtGui/QMainWindow>
#include "ui_gamma_gui.h"
#include <QStringList>
#include <vector>

//#include "itkImageFileReader.h"
//#include "itkImageFileWriter.h"
#include "itk_image_type.h"

//#include "YK16GrayImage.h"
//#include <fstream>

using namespace std;

class Gamma_parms;

class DlgGammaView;
class YK16GrayImage;
class qyklabel;
class QStandardItemModel;

class gamma_gui : public QMainWindow
{
    Q_OBJECT

public:
    gamma_gui(QWidget *parent = 0, Qt::WFlags flags = 0);
    ~gamma_gui(); 
    
    QString GammaMain(Gamma_parms* parms);     

    void Load_FilesToMem();//all ref, comp, gamma map should be prepared    
    void UpdateProbePos(qyklabel* qlabel);
    

    //void UpdateTable();
    void UpdateTable(vector<QPointF>& vData1, vector<QPointF>& vData2, vector<QPointF>& vData3, float fMag);    
    

    public slots:        
        void SLT_Load_RD_Ref();
        void SLT_Load_RD_Comp();        
        void SLT_LoadResults();

        void SLT_RunBatchGamma();
        void SLT_ProfileView();

        void SLT_DrawDoseImages();// refer to probe positions, selected 3D file (spPointer), plane direction
        void SLT_DrawGammaMap3D();
        void SLT_DrawGammaMap2D();
        void SLT_UpdateComboContents();        

        void SLT_DrawAll();
        void SLT_DrawTable();
        void SLT_DrawChart();

        void SLT_UpdateReportTxt();

        void SLT_UpdateProbePosRef();
        void SLT_UpdateProbePosComp();
        void SLT_UpdateProbePosGamma2D();
        void SLT_UpdateProbePosGamma3D();
        void SLT_CopyTableToClipboard();
        //void SLT_DrawGraph(); 
        void SLT_DrawGraph(bool bInitMinMax = false);

        

public:    
    QStringList m_strlistPath_RD_Ref;
    QStringList m_strlistPath_RD_Comp;

    QStringList m_strlistFileBaseName_Ref;
    QStringList m_strlistFileBaseName_Comp;
    
    QStringList m_strlistBatchReport;
    
    QStringList m_strlistPath_Output_Gammamap;
    QStringList m_strlistPath_Output_Failure;
    QStringList m_strlistPath_Output_Report;

    //DlgGammaView* m_pView;    

    vector<FloatImageType::Pointer> m_vRefDoseImages;
    vector<FloatImageType::Pointer> m_vCompDoseImages;
    vector<FloatImageType::Pointer> m_vGammaMapImages;

    /*FloatImageType::Pointer m_spRefDoseImages;
    FloatImageType::Pointer m_spCompDoseImages;
    FloatImageType::Pointer m_spGammaMapImages;*/

    YK16GrayImage* m_pCurImageRef;
    YK16GrayImage* m_pCurImageComp;
    YK16GrayImage* m_pCurImageGamma3D;
    YK16GrayImage* m_pCurImageGamma2D;

    QStandardItemModel *m_pTableModel;
private:
    Ui::gamma_guiClass ui;
};

#endif // gamma_gui_H
