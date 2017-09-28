#include <algorithm>
#include <fstream>
#include <QFileDialog>
#include <QMessageBox>
#include <QStringList>
#include <QPainter>
#include "badpixeldetector.h"
#include "YK16GrayImage.h"

using namespace std;


bool CompareByXVal(BADPIXELMAP first, BADPIXELMAP second)
{
    if (first.BadPixX != second.BadPixX)
    {
        return first.BadPixX < second.BadPixX; //ascending
    }
    else
    {
        return first.BadPixY < second.BadPixY; //ascending
    }

}

bool CompareByYVal(BADPIXELMAP first, BADPIXELMAP second)
{
    if (first.BadPixY != second.BadPixY)
    {
        return first.BadPixY < second.BadPixY;
    }
    else
    {
        return first.BadPixX < second.BadPixX;
    }
}


bool CheckSame(BADPIXELMAP first, BADPIXELMAP second)
{
    // Only check bad pixel position.  If you check replacement, multiple
    // gain images cannot be combined.
    if (first.BadPixX == second.BadPixX && first.BadPixY == second.BadPixY)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool CompareImageMean (
    YK16GrayImage* first, YK16GrayImage* second)
{
    double mean1;
    double SD1;
    double min1;
    double max1;

    double mean2;
    double SD2;
    double min2;
    double max2;

    first->CalcImageInfo(mean1,SD1,min1,max1);
    second->CalcImageInfo(mean2,SD2,min2,max2);

    if (mean1 > mean2)
        return false;
    else
        return true;
}

QPoint gGetMedianIndex (
    int X, int Y, int iMedianSize, int width, int height,
    unsigned short* pSrcImage)
{
    QPoint tmpResult = QPoint(X,Y);

    if (pSrcImage == NULL)
        return tmpResult;

    int imgSize = width*height;

    // Force iMedianSize to be an odd number.  Set arm to the half-size.
    int size = ((iMedianSize-1)/2)*2+1;
    int arm = size / 2.0;

    // if the point is near an edge, do not replace
    if (X-arm < 0 || X+arm > width-1 || Y-arm < 0 || Y+arm > height-1) {
        return tmpResult;
    }

    int bufSize = size*size;
    PIXINFO* medianPixBuf = new PIXINFO [bufSize];
    int cnt = 0;
    int i, j;
    for (i = -arm ; i <= arm; i++) {
        for (j = -arm ; j <= arm; j++) {
            medianPixBuf[cnt].pixValue = pSrcImage[(Y+i)*width + (X+j)];
            medianPixBuf[cnt].infoX = X+j;
            medianPixBuf[cnt].infoY = Y+i;
            cnt++;
        }
    }

    // GCS: This algorithm could choose a bad pixel for replacement
    for (i = 0 ; i<bufSize-1 ; i++) {
        for (j = i+1 ; j<bufSize ; j++) {
            if (medianPixBuf[i].pixValue > medianPixBuf[j].pixValue) {
                PIXINFO tmp;
                tmp.infoX = 0;
                tmp.infoY = 0;
                tmp.pixValue = 0;

               tmp = medianPixBuf[i];
                medianPixBuf[i] = medianPixBuf[j];
                medianPixBuf[j] = tmp;
            }
        }
    }

    int medianIndex = bufSize / 2.0;

    tmpResult.setX(medianPixBuf[medianIndex].infoX);
    tmpResult.setY(medianPixBuf[medianIndex].infoY);

    delete [] medianPixBuf;

    return tmpResult;
}

BadPixelDetector::BadPixelDetector (QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags)
{
    ui.setupUi(this);

    m_iWidth = 2304;
    m_iHeight = 3200;

    m_pImageYKDark = new YK16GrayImage(2304, 3200);
    m_pImageYKGain = new YK16GrayImage(2304, 3200);
    m_fPercentThre = 30.0; //30% increase is mandatory
    m_iMedianSize = 3;
}

BadPixelDetector::~BadPixelDetector()
{
    delete m_pImageYKDark;
    delete m_pImageYKGain;
}

void BadPixelDetector::SLT_LoadDarkImage()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Raw image file (*.raw)", 0,0);

    if (!m_pImageYKDark->LoadRawImage(fileName.toLocal8Bit().constData(),m_iWidth,m_iHeight))
        return;

    m_strSrcFilePathDark = fileName;

    double mean;
    double SD;
    double max;
    double min;
    m_pImageYKDark->CalcImageInfo(mean, SD, max, min);

    ui.sliderDarkMin->setValue((int)(mean - 4*SD));
    ui.sliderDarkMax->setValue((int)(mean + 4*SD));

    SLT_DrawDarkImage();

}

void BadPixelDetector::SLT_LoadGainImage()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Raw image file (*.raw)", 0,0);

    if (!m_pImageYKGain->LoadRawImage(fileName.toLocal8Bit().constData(),m_iWidth,m_iHeight))
        return;

    m_strSrcFilePathGain = fileName;

    double mean;
    double SD;
    double max;
    double min;
    m_pImageYKGain->CalcImageInfo(mean, SD, max, min);

    ui.sliderGainMin->setValue((int)(mean - 4*SD));
    ui.sliderGainMax->setValue((int)(mean + 4*SD));

    SLT_DrawGainImage();

}

void BadPixelDetector::SLT_DrawDarkImage()
{
    if (m_pImageYKDark->IsEmpty())
        return;

    m_pImageYKDark->FillPixMapMinMax(ui.sliderDarkMin->value(), ui.sliderDarkMax->value());
    //m_pImageYKDark->DrawToLabel(ui.labelImageDark);

    ui.labelImageDark->SetBaseImage(m_pImageYKDark);
    ui.labelImageDark->update();
}

void BadPixelDetector::SLT_DrawGainImage()
{
    if (m_pImageYKGain->IsEmpty())
        return;

    m_pImageYKGain->FillPixMapMinMax(ui.sliderGainMin->value(), ui.sliderGainMax->value());
    //m_pImageYKGain->DrawToLabel(ui.labelImageGain);

    ui.labelImageGain->SetBaseImage(m_pImageYKGain);
    ui.labelImageGain->update();
}

void BadPixelDetector::SLT_ShowBadPixels() //copy defect points and update both dsp
{
    vector<QPoint> vTmpData;

    vector<BADPIXELMAP>::iterator it;

    for (it = m_vPixelReplMap.begin() ; it != m_vPixelReplMap.end() ; it++)
    {
        QPoint tmpPt;
        tmpPt.setX((*it).BadPixX);
        tmpPt.setY((*it).BadPixY);
        vTmpData.push_back(tmpPt);
    }

    ui.labelImageDark->ConvertAndCopyPoints(vTmpData, m_iWidth, m_iHeight);
    ui.labelImageGain->ConvertAndCopyPoints(vTmpData, m_iWidth, m_iHeight);
    ui.labelImageDark->update();
    ui.labelImageGain->update();
}

void BadPixelDetector::SLT_SavePixelMap()
{
    if (m_vPixelReplMap.empty())
        return;

    QString fileName = QFileDialog::getSaveFileName(this, "Save Pixel Map", "", "point mapping file (*.pmf)",0,0);

    ofstream fout;

    fout.open(fileName.toLocal8Bit().constData());

    fout << "#ACQUIRE4030E_BADPIXEL_MAP" << endl;
    fout << "#Src Dark File:" << "	" << m_strSrcFilePathDark.toLocal8Bit().constData() << endl;
    fout << "#Src Gain File:" << "	" << m_strSrcFilePathGain.toLocal8Bit().constData() << endl;
    fout << "#Total map size: " << "	" << m_vPixelReplMap.size() << endl;
    fout << "#ORIGINAL_X" << "	" << "ORIGINAL_Y" << "	" << "SUBSTITUTE_X" << "	" << "SUBSTITUTE_Y" << endl;

    vector<BADPIXELMAP>::iterator it;

    for (it = m_vPixelReplMap.begin() ; it != m_vPixelReplMap.end() ; it++)
    {
        BADPIXELMAP tmpMap;
        tmpMap = (*it);
        fout << tmpMap.BadPixX << "	" << tmpMap.BadPixY << "	" << tmpMap.ReplPixX << "	" << tmpMap.ReplPixY << endl;
    }
    fout.close();
}


void BadPixelDetector::SLT_UncorrectDark() //Gain Image + Dark (when gain image was already added by Dark image
{
    int size = m_iWidth*m_iHeight;

    for (int i = 0 ; i<size ; i++)
    {
        int tmpVal = (int)(this->m_pImageYKGain->m_pData[i] + this->m_pImageYKDark->m_pData[i]);

        if (tmpVal > 65535)
            m_pImageYKGain->m_pData[i] = 0;
        else
            m_pImageYKGain->m_pData[i] = (unsigned short)tmpVal;
    }
}

void BadPixelDetector::SLT_DoReplacement_Dark()
{
    if (m_vPixelReplMap.empty())
        return;

    int oriIdx, replIdx;

    vector<BADPIXELMAP>::iterator it;

    for (it = m_vPixelReplMap.begin() ; it != m_vPixelReplMap.end(); it++)
    {
        BADPIXELMAP tmpData= (*it);
        oriIdx = tmpData.BadPixY * m_iWidth + tmpData.BadPixX;
        replIdx = tmpData.ReplPixY * m_iWidth + tmpData.ReplPixX;
        m_pImageYKDark->m_pData[oriIdx] = m_pImageYKDark->m_pData[replIdx];
    }

    SLT_DrawDarkImage();
}

void BadPixelDetector::SLT_DoReplacement_Gain()
{
    if (m_vPixelReplMap.empty())
        return;

    int oriIdx, replIdx;

    vector<BADPIXELMAP>::iterator it;

    for (it = m_vPixelReplMap.begin() ; it != m_vPixelReplMap.end(); it++)
    {
        BADPIXELMAP tmpData= (*it);

        oriIdx = tmpData.BadPixY * m_iWidth + tmpData.BadPixX;
        replIdx = tmpData.ReplPixY * m_iWidth + tmpData.ReplPixX;
        m_pImageYKGain->m_pData[oriIdx] = m_pImageYKGain->m_pData[replIdx];
    }

    SLT_DrawGainImage();
}

void BadPixelDetector::LoadBadPixelMap(const char* filePath)
{
    m_vPixelReplMap.clear();

    ifstream fin;
    fin.open(filePath);

    if (fin.fail())
        return;

#define MAX_LINE_LENGTH 1024
    char str[MAX_LINE_LENGTH];
    while (!fin.eof())
    {
        memset(str, 0, MAX_LINE_LENGTH);
        fin.getline(str, MAX_LINE_LENGTH);
        QString tmpStr = QString(str);

        if (tmpStr.contains("#ORIGINAL_X"))
            break;
    }

    while (!fin.eof())
    {
        memset(str, 0, MAX_LINE_LENGTH);
        fin.getline(str, MAX_LINE_LENGTH);
        QString tmpStr = QString(str);

        QStringList strList = tmpStr.split("	");

        if (strList.size() == 4)
        {
            BADPIXELMAP tmpData;
            tmpData.BadPixX = strList.at(0).toInt();
            tmpData.BadPixY = strList.at(1).toInt();
            tmpData.ReplPixX = strList.at(2).toInt();
            tmpData.ReplPixY = strList.at(3).toInt();
            m_vPixelReplMap.push_back(tmpData);
        }
    }

    fin.close();
}

void BadPixelDetector::SLT_LoadBadPixelMap()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open pixel map file", "", "point mapping file (*.txt *.pmf)", 0,0);
    LoadBadPixelMap(fileName.toLocal8Bit().constData());

    SLT_ShowBadPixels();
}

void BadPixelDetector::SLT_SaveCurDark()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save Current Dark Image", "", "Raw Image File (*.raw)",0,0);

    if (fileName.length() > 3)
    {
        m_pImageYKDark->SaveDataAsRaw(fileName.toLocal8Bit().constData());
    }
}

void BadPixelDetector::SLT_SaveCurGain()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save Current Dark Image", "", "Raw Image File (*.raw)",0,0);

    if (fileName.length() > 3)
    {
        m_pImageYKGain->SaveDataAsRaw(fileName.toLocal8Bit().constData());
    }
}


//called before median index input
int BadPixelDetector::AddBadPixLine (
    vector<BADPIXELMAP>& vPixelReplMap, int direction)
{
    if (vPixelReplMap.empty())
        return 0;

    int oldCnt = vPixelReplMap.size();

    int result = 0;

    if (direction == 0)
    {
        //should be implemented later.. now all of the bad pixels
        // are generated along vertical direction
    }
    else if (direction == 1) //vertical line
    {
        sort(vPixelReplMap.begin(),vPixelReplMap.end(),CompareByXVal);

        vector<BADPIXELMAP>::iterator it;
        vector<int> vSameXCnt;
        vSameXCnt.resize (m_iWidth, 0);

        // Count bad pixels in each column
        for (it = vPixelReplMap.begin() ; it != vPixelReplMap.end(); it++)
        {
            if ((*it).BadPixX < 0 || (*it).BadPixX >= m_iWidth) {
                printf ("Bogus entry: %d\n", (*it).BadPixX);
            }
            vSameXCnt[(*it).BadPixX]++;
        }

        int badLinePercent = ui.lineEditBadLineDefPerc->text().toInt();

        // Find columns with bad pixels that exceed threshold
        list<int> badColumns;
        vector<int>::iterator ptIt;
        int col = 0;
        for (ptIt = vSameXCnt.begin(); ptIt != vSameXCnt.end() ;++ptIt)
        {
            if ((*ptIt) >= m_iHeight * badLinePercent / 100.0)
            {
                printf ("Bad column %d (%d/%d)\n", col, *ptIt, m_iHeight);
                badColumns.push_back (*ptIt);
            }
            col++;
        }

        // For each bad column, add bad pixels to map
        for (list<int>::iterator it = badColumns.begin();
             it != badColumns.end(); it++)
        {
            int col = *it;
            for (int i = 0 ; i < m_iHeight ;i++)
            {
                BADPIXELMAP tmpData;
                tmpData.BadPixX = col;
                tmpData.BadPixY = i;
                tmpData.ReplPixX = -1;
                tmpData.ReplPixY = -1;
                vPixelReplMap.push_back(tmpData);
            }
        }

        // Sort pixel map
        sort(vPixelReplMap.begin(),vPixelReplMap.end(),CompareByXVal);
        printf("current size before duplication removal = %d\n", vPixelReplMap.size());

        // Remove duplicates
        vPixelReplMap.erase(unique(vPixelReplMap.begin(), vPixelReplMap.end(), CheckSame), vPixelReplMap.end());
        printf("current size after duplication removal = %d\n", vPixelReplMap.size());
    }
    else
    {
        return 0;
    }

    int newCnt = vPixelReplMap.size();

    return (newCnt - oldCnt);
}

void BadPixelDetector::SLT_DetectBadPixels()
{
    printf("[1] Current vector size = %d\n", m_vPixelReplMap.size());
    DetectBadPixels (true);
    printf("[3] Current vector size = %d\n", m_vPixelReplMap.size());
    SLT_ShowBadPixels();
}


void BadPixelDetector::SLT_AccumulateBadPixels()
{
    DetectBadPixels (false);
    printf("Current vector size = %d\n", m_vPixelReplMap.size());
    SLT_ShowBadPixels();
}

void BadPixelDetector::SLT_AddManual()
{
    QString s = ui.lineEditManualAdd->text();
    QStringList sl = s.split (" ,\"", QString::SkipEmptyParts);

    size_t numbers_read = 0;
    bool ok = true;
    int manual_add[2];
    QStringList::const_iterator it = sl.constBegin();
    while (it != sl.constEnd()) {
        int x = (*it).toInt(&ok);
        if (!ok) {
            break;
        }
        manual_add[numbers_read] = x;
        ++numbers_read;
        if (numbers_read > 2) {
            break;
        }
        ++it;
    }

    if (!ok || numbers_read == 0 || numbers_read > 2) {
        QMessageBox::information (this, "Bad pixel detector",
            QString ("Please specify pixel or column as \"row col\" or \"col\" \n"));
    }

    if (numbers_read == 1) {
        QMessageBox::information (this, "Bad pixel detector",
            QString ("Bad column %1").arg(manual_add[0]));
    } else {
        QMessageBox::information (this, "Bad pixel detector",
            QString ("Bad pixel %1 %2").arg(manual_add[0],manual_add[1]));
    }
//    DetectBadPixels(m_vPixelReplMap, true);
//    printf("Current vector size = %d\n", m_vPixelReplMap.size());
//    SLT_ShowBadPixels();
}

void BadPixelDetector::DetectBadPixels (bool bRefresh)
{
    if (m_pImageYKDark->IsEmpty() || m_pImageYKGain->IsEmpty())
        return;

    int i = 0;
    int j = 0;

    QString str = ui.lineEditPercentThre->text();
    m_fPercentThre = str.toDouble();

    QString str2 = ui.lineEditMedianSize->text();
    m_iMedianSize = str2.toDouble();

    int imgSize = m_iWidth * m_iHeight;

    double meanDark;
    double SDDark;
    double minDark;
    double maxDark;
    m_pImageYKDark->CalcImageInfo(meanDark, SDDark, minDark, maxDark);

    double meanGain;
    double SDGain;
    double minGain;
    double maxGain;
    m_pImageYKGain->CalcImageInfo(meanGain, SDGain, minGain, maxGain);

    vector<BADPIXELMAP> vTmpVec;
    printf("[2o] Current vector size = %d\n", vTmpVec.size());
    
    // minimum percent of average increase in pixel value
    double diffThreshold = (meanGain - meanDark) * m_fPercentThre / 100.0;
    printf("diffThresh = %f (%f, %f)\n", diffThreshold, meanGain, meanDark);

    for (i = 0; i < m_iHeight ; i++) {
        for (j = 0; j < m_iWidth ; j++) {
            BADPIXELMAP pixMap;
            pixMap.BadPixX = -1;
            pixMap.BadPixY = -1;
            pixMap.ReplPixX = -1;
            pixMap.ReplPixY = -1;

            // forget about first line
            // GCS: Why?
            if (i == 0) {
                continue;
            }

            size_t pixel_index = m_iWidth*i + j;
            unsigned short dark_val = m_pImageYKDark->m_pData[pixel_index];
            unsigned short gain_val = m_pImageYKGain->m_pData[pixel_index];
            
            // if gain pixel value is not at least threshold greater than
            // dark pixel value, mark as bad
            if (gain_val < dark_val + diffThreshold)
            {
                pixMap.BadPixX = j;
                pixMap.BadPixY = i;
                vTmpVec.push_back(pixMap);
            }
        }
    }
    printf("[2a] Current vector size = %d\n", vTmpVec.size());

    // Audit bad pixels. if DEFAULT_PERCENT_BADPIX_ON_COLUMN (e.g.60%)
    // pixels on one line are badpixels, make that column as bad pixel column
    // direction 0 = hor (row), dir 1 = ver(column)
    int addedCnt = AddBadPixLine(vTmpVec, 1);

    printf("[2b] Current vector size = %d\n", vTmpVec.size());
    
    vector<BADPIXELMAP>::iterator it;
    for (it = vTmpVec.begin(); it != vTmpVec.end() ; it++) {
        int srcX = (*it).BadPixX;
        int srcY = (*it).BadPixY;

        QPoint tmpPt;
        tmpPt = gGetMedianIndex (srcX, srcY, m_iMedianSize, m_iWidth,
            m_iHeight, m_pImageYKGain->m_pData);

        (*it).ReplPixX = tmpPt.x();
        (*it).ReplPixY = tmpPt.y();
    }

    printf("[2c] Current vector size = %d\n", vTmpVec.size());
    
    if (bRefresh) {
        m_vPixelReplMap.clear();
    }

    printf("[2d] Current vector size = %d\n", m_vPixelReplMap.size());
    
    for (it = vTmpVec.begin(); it != vTmpVec.end() ; it++) {
        m_vPixelReplMap.push_back (*it);
    }

    printf("[2e] Current vector size = %d\n", m_vPixelReplMap.size());
    
    if (!bRefresh) {
        // accumulation mode, therefore duplicate entries should be deleted
        sort (m_vPixelReplMap.begin(),m_vPixelReplMap.end(),CompareByXVal);
        m_vPixelReplMap.erase (unique(m_vPixelReplMap.begin(),
                m_vPixelReplMap.end(), CheckSame), m_vPixelReplMap.end());
    }

    printf("[2f] Current vector size = %d\n", m_vPixelReplMap.size());
    
    SLT_DrawGainImage();
}
