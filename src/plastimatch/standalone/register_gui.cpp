#include "register_gui.h"
#include <QString>
#include <QFileDialog>
#include <QListView>
#include <QMessageBox>
#include <fstream>

#include "mha_io.h"
#include "nki_io.h"
#include "plm_image.h"
#include "rt_study_metadata.h"

//added for register_gui
#include <QFileInfo>

#include "gamma_dose_comparison.h"
#include "logfile.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "plm_math.h"
#include "pcmd_gamma.h"
#include "print_and_exit.h"

#include "plm_file_format.h"
#include "rt_study.h"


register_gui::register_gui(QWidget *parent, Qt::WFlags flags)
: QMainWindow(parent, flags)
{
    ui.setupUi(this);
}

register_gui::~register_gui()
{ 

}

void register_gui::SLT_Load_RD_Ref()
{
    QStringList tmpList = QFileDialog::getOpenFileNames(this, "Select one or more files to open", "/home", "3D dose file (*.dcm *.mha)");

    int iFileCnt = tmpList.size();

    if (iFileCnt < 1)
        return;

    ui.plainTextEdit_RD_Ref->clear();
    m_strlistPath_RD_Ref.clear();
    m_strlistFileBaseName_Ref.clear();

    m_strlistPath_RD_Ref = tmpList;

    for (int i = 0; i < iFileCnt; i++)
    {
        ui.plainTextEdit_RD_Ref->appendPlainText(m_strlistPath_RD_Ref.at(i)); //just for display

        QFileInfo tmpInfo = QFileInfo(m_strlistPath_RD_Ref.at(i));

        m_strlistFileBaseName_Ref.push_back(tmpInfo.completeBaseName());
    }
}

void register_gui::SLT_Load_RD_Comp()
{
    QStringList tmpList = QFileDialog::getOpenFileNames(this, "Select one or more files to open", "/home", "3D dose file (*.dcm *.mha)");

    int iFileCnt = tmpList.size();

    if (iFileCnt < 1)
        return;
    
    ui.plainTextEdit_RD_Comp->clear();
    m_strlistPath_RD_Comp.clear();
    m_strlistFileBaseName_Comp.clear();

    m_strlistPath_RD_Comp = tmpList;

    for (int i = 0; i < iFileCnt; i++)
    {
        ui.plainTextEdit_RD_Comp->appendPlainText(m_strlistPath_RD_Comp.at(i)); //just for display

        QFileInfo tmpInfo = QFileInfo(m_strlistPath_RD_Comp.at(i));

        m_strlistFileBaseName_Comp.push_back(tmpInfo.completeBaseName());
    }    
}