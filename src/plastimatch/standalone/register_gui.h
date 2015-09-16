#ifndef REGISTER_GUI_H
#define REGISTER_GUI_H

#include <QtGui/QMainWindow>
#include "ui_register_gui.h"
#include <QStringList>
#include <vector>


using namespace std;

class register_gui : public QMainWindow
{
    Q_OBJECT

public:
    register_gui(QWidget *parent = 0, Qt::WFlags flags = 0);
    ~register_gui();    
    

    public slots:        
        void SLT_Load_RD_Ref();
        void SLT_Load_RD_Comp();

public:    
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

#endif
