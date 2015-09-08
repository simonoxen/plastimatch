#include "register_gui.h"
#include <QString>
#include <QFileDialog>
#include <QListView>
#include <QMessageBox>
//#include "YK16GrayImage.h"
#include <fstream>

#include "mha_io.h"
#include "nki_io.h"
//#include "volume.h"
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
//#include <fstream>

register_gui::register_gui(QWidget *parent, Qt::WFlags flags)
: QMainWindow(parent, flags)
{
    ui.setupUi(this);
    //m_pImgOffset = NULL;
    //m_pImgGain = NULL;
    ////Badpixmap;
    //m_pImgOffset = new YK16GrayImage(IMG_WIDTH, IMG_HEIGHT);
    //m_pImgGain = new YK16GrayImage(IMG_WIDTH, IMG_HEIGHT);

    //	const char* inFileName = "C:\\test.scan";
    //	const char* outFileName = "C:\\test.mha";


    /*Volume *v = nki_load (inFileName);
    if (!v)
    {
    printf("file reading error\n");		
    }
    write_mha(outFileName, v);*/


}

register_gui::~register_gui()
{
    //delete m_pImgOffset;
    //delete m_pImgGain;

    //m_vPixelReplMap.clear(); //not necessary

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

void register_gui::SLT_RunBatchGamma()
{

    //number of batch should be matched
    int cntRef = m_strlistPath_RD_Ref.count();
    int cntComp = m_strlistPath_RD_Comp.count();


    if (cntRef*cntComp == 0 || cntRef != cntComp)
    {
        cout << "ERROR! Invalid input file counts" << endl;
        return;
    }

    m_strlistBatchReport.clear();
    m_strlistPath_Output_Gammamap.clear();
    m_strlistPath_Output_Failure.clear();
    m_strlistPath_Output_Report.clear();

    for (int i = 0; i < cntRef; i++)
    {
        QString strPathRef = m_strlistPath_RD_Ref.at(i);
        QString strPathComp = m_strlistPath_RD_Comp.at(i);

        if (strPathRef.length() < 2 || strPathComp.length() < 2)
            continue;//skip this pair

        Gamma_parms parms;
        //Gamma param: should come from the UI

        parms.mask_image_fn = "";
        //parms->reference_dose;
        parms.gamma_max = 2.0;
        parms.b_compute_full_region = false;
        parms.b_resample_nn = false; //default: false

        //From File List
        parms.ref_image_fn = strPathRef.toLocal8Bit().constData();
        parms.cmp_image_fn = strPathComp.toLocal8Bit().constData();

        //From GUI
        if (ui.checkBox_inhereResample->isChecked())
            parms.f_inherent_resample_mm = ui.lineEdit_inhereResample->text().toDouble();
        else
            parms.f_inherent_resample_mm = -1.0;
        
        parms.b_interp_search = ui.checkBox_Interp_search->isChecked();
        
        if (ui.radioButton_localGamma->isChecked())
        {
            parms.b_local_gamma = true;
            parms.reference_dose = 0.0;
        }
        else
        {
            parms.b_local_gamma = false;
            parms.reference_dose = ui.lineEdit_refDoseInGy->text().toDouble();
        }

        parms.dta_tolerance = ui.lineEdit_dta_tol->text().toDouble();
        parms.dose_tolerance = ui.lineEdit_dose_tol->text().toDouble() / 100.0;//gui input: 3% --> param: 0.03
        parms.f_analysis_threshold = ui.lineEdit_cutoff_dose->text().toDouble() / 100.0;
        
        //Saving folder: comp folder. FileName Should Include dta, dose, local/global

        QFileInfo fInfo = QFileInfo(strPathComp);
        QString dirPath = fInfo.absolutePath();
        QString baseName = fInfo.completeBaseName();

        
 //       m_strlistPath_Output_Gammamap.clear();
   //     m_strlistPath_Output_Failure.clear();
     //   m_strlistPath_Output_Report.clear();

        QString strLocGlob;

        if (parms.b_local_gamma)
            strLocGlob = "loc";        
        else
            strLocGlob = "glb";

        QString strSettingAbs = QString::number(parms.dta_tolerance, 'f', 0) + "mm_" + ""
            + QString::number(parms.dose_tolerance*100.0, 'f', 0) + "%_" + strLocGlob;


        if (ui.checkBox_gammamap_output->isChecked())
        {            
            QString outputPath = dirPath + baseName + "_gammamap_" + strSettingAbs + ".mha";
            parms.out_image_fn = outputPath.toLocal8Bit().constData();
            m_strlistPath_Output_Gammamap.push_back(outputPath);
        }

        if (ui.checkBox_failuremap_output->isChecked())
        {
            QString outputPath = dirPath + baseName + "_failmap_" + strSettingAbs + ".mha";
            parms.out_failmap_fn = outputPath.toLocal8Bit().constData();
            m_strlistPath_Output_Failure.push_back(outputPath);            
        }           

        QString outputPath = dirPath + baseName + "_report_" + strSettingAbs + ".txt";
        parms.out_report_fn = outputPath.toLocal8Bit().constData();
        m_strlistPath_Output_Report.push_back(outputPath);

        QString overallReport = GammaMain(&parms);
        m_strlistBatchReport.push_back(overallReport);
    }

    //After the batch mode analysis, export the simpe report.    
    //Only when the number of files is > 1
    /*if (cntRef == 1)
        return;*/
        
    QString fileName = QFileDialog::getSaveFileName(this, "Save batch report file", "", "report (*.txt)", 0, 0);

    if (fileName.length() < 1)
        return;

    ofstream fout;
    fout.open(fileName.toLocal8Bit().constData());
    fout << "Reference_File\t"
        << "Compare_File\t"
        << "dta_tolerance[mm]\t"
        << "dose_tolerance[%]\t"
        << "doseCutoff[%]\t"
        << "Local/Global\t"
        << "Ref_dose[Gy]\t"
        << "VoxNumAnalyzed\t"
        << "VoxNumPassed\t"
        << "GammaPassRate[%]" << endl;

    for (int i = 0; i < cntRef; i++)
    {
        fout << m_strlistBatchReport.at(i).toLocal8Bit().constData() << endl;
    }

    fout.close();
}

QString register_gui::GammaMain(Gamma_parms* parms)
{
    QString reportResult;
    Gamma_dose_comparison gdc;

    //DICOM_RD compatible (added by YK, Feb 2015)
    //In the prev version, RD couldn't be read directly due to the scale factor inside of the DICOM file.
    //work-around was to use (plastimatch convert ...)
    //here, that feature has been integrated into plastimatch gamma
    Plm_file_format file_type_ref, file_type_comp;
    Rt_study rt_study_ref, rt_study_comp;

    file_type_ref = plm_file_format_deduce(parms->ref_image_fn.c_str());
    file_type_comp = plm_file_format_deduce(parms->cmp_image_fn.c_str());

    if (file_type_ref == PLM_FILE_FMT_DICOM_DOSE) {
        rt_study_ref.load(parms->ref_image_fn.c_str(), file_type_ref);
        if (rt_study_ref.has_dose()){
            gdc.set_reference_image(rt_study_ref.get_dose()->clone());
        }
        else{
            gdc.set_reference_image(parms->ref_image_fn.c_str());
        }
    }
    else {
        gdc.set_reference_image(parms->ref_image_fn.c_str());
    }

    if (file_type_comp == PLM_FILE_FMT_DICOM_DOSE) {
        rt_study_comp.load(parms->cmp_image_fn.c_str(), file_type_comp);
        if (rt_study_comp.has_dose()) {
            gdc.set_compare_image(rt_study_comp.get_dose()->clone());
        }
        else {
            gdc.set_compare_image(parms->cmp_image_fn.c_str());
        }

    }
    else {
        gdc.set_compare_image(parms->cmp_image_fn.c_str());
    }
    //End DICOM-RD    

    if (parms->mask_image_fn != "") {
        gdc.set_mask_image(parms->mask_image_fn);
    }

    gdc.set_spatial_tolerance(parms->dta_tolerance);
    gdc.set_dose_difference_tolerance(parms->dose_tolerance);
    if (parms->have_reference_dose) {
        gdc.set_reference_dose(parms->reference_dose);
    }
    gdc.set_gamma_max(parms->gamma_max);

    /*Extended by YK*/
    gdc.set_interp_search(parms->b_interp_search);//default: false
    gdc.set_local_gamma(parms->b_local_gamma);//default: false
    gdc.set_compute_full_region(parms->b_compute_full_region);//default: false
    gdc.set_resample_nn(parms->b_resample_nn); //default: false

    if (parms->f_inherent_resample_mm > 0.0) {
        gdc.set_inherent_resample_mm(parms->f_inherent_resample_mm);
    }

    if (parms->f_analysis_threshold > 0) {
        gdc.set_analysis_threshold(parms->f_analysis_threshold);//0.1 = 10%
    }

    gdc.run();

    if (parms->out_image_fn != "") {
        Plm_image::Pointer gamma_image = gdc.get_gamma_image();
        gamma_image->save_image(parms->out_image_fn);
    }

    if (parms->out_failmap_fn != "") {
        gdc.get_fail_image()->save_image(parms->out_failmap_fn);
    }

    if (parms->out_report_fn != "") {
        //Export utput text using gdc.get_report_string();
        std::ofstream fout;
        fout.open(parms->out_report_fn.c_str());
        if (!fout.fail()){
            fout << "Reference_file_name\t" << parms->ref_image_fn.c_str() << std::endl;
            fout << "Compare_file_name\t" << parms->cmp_image_fn.c_str() << std::endl;

            fout << gdc.get_report_string();
            fout.close();
        }
    }
    printf ("Pass rate = %2.6f %%\n", gdc.get_pass_fraction() * 100.0);
    
    //Composite a result string
    //FileName
    QFileInfo info_ref = QFileInfo(parms->ref_image_fn.c_str());
    QFileInfo info_comp = QFileInfo(parms->cmp_image_fn.c_str());   
    
    QString fileName_ref = info_ref.fileName();
    QString fileName_comp = info_comp.fileName();

    QString strTol_dta = QString::number(parms->dta_tolerance,'f',2);
    QString strTol_dose = QString::number(parms->dose_tolerance*100.0, 'f', 0);
    QString strDoseCutoff = QString::number(parms->f_analysis_threshold*100, 'f', 0);

    QString strLocalOrGlobal;
    if (parms->b_local_gamma)
        strLocalOrGlobal = "local";
    else
        strLocalOrGlobal = "global";

    //if this is dcm, save the mha files
    if (info_ref.suffix() == "dcm" || info_ref.suffix() == "DCM")
    {
        QString newPath = info_ref.absolutePath() + "\\" + info_ref.completeBaseName() + ".mha";
        Plm_image* pImg = gdc.get_ref_image();
        pImg->save_image(newPath.toLocal8Bit().constData());
    }

    if (info_comp.suffix() == "dcm" || info_comp.suffix() == "DCM")
    {
        QString newPath = info_comp.absolutePath() + "\\" + info_comp.completeBaseName() + ".mha";
        Plm_image* pImg = gdc.get_comp_image();
        pImg->save_image(newPath.toLocal8Bit().constData());
    }

    QString strRef_dose = QString::number(gdc.get_reference_dose(), 'f', 2);//Gy
    QString strVoxNumAnalyzed = QString::number(gdc.get_analysis_num_vox(), 'f', 2);
    QString strVoxNumPassed = QString::number(gdc.get_passed_num_vox(), 'f', 2);
    QString strPassRate = QString::number(gdc.get_pass_fraction()*100.0, 'f', 2);

    reportResult = fileName_ref + "\t"
        + fileName_comp + "\t"
        + strTol_dta + "\t"
        + strTol_dose + "\t"
        + strDoseCutoff + "\t"
        + strLocalOrGlobal + "\t"
        + strRef_dose + "\t"
        + strVoxNumAnalyzed + "\t"
        + strVoxNumPassed + "\t"
        + strPassRate;

    return reportResult;
}