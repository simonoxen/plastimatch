#include <QFileDialog>
#include <math.h>
#include <vector>
#include <QMessageBox>
#include "CMatrix.h"
#include <QProgressDialog>
#include <algorithm>
#include "crossremoval.h"
#include "YK16GrayImage.h"
#include "YK16GrayImageITK.h"

using namespace std;

//void g_GetEq(QPoint* ptArr, int arrSize, int degree, double* coeff)
void g_GetEq(vector<QPoint>& vPt, int degree, double* coeff)
{
    int arrSize = vPt.size();

    if (arrSize < 2)
        return;


    int i, j;
    CMatrix dataA(degree+1, arrSize, 0);// 3 x 2 행렬 
    CMatrix dataB(1, arrSize, 0); // 3 x 1 행렬 	

    for (i = 0 ; i < degree + 1 ; i++)
    {
        for (j=0 ; j<arrSize ; j++)
        {
            dataA[i][j] = pow(vPt.at(j).x(), (double)(degree- i));
        }	
    }

    for (j=0 ; j<arrSize ; j++)
    {
        dataB[0][j] = vPt.at(j).y();
    }	

    CMatrix result (1, degree+1 , 0); //result는 2 by 1

    CMatrix inv(degree+1, degree+1, 0); //2 by 2 행렬 	
    (dataA*(!dataA)).Inverse(inv);

    result = (dataB*((!dataA)*inv));


    //(dataA*(!dataA)).Save_File("C:\\A_A_T.txt");
    //dataA.Save_File("C:\\MatrixA.txt");
    //(!dataA).Save_File("C:\\MatrixA_T.txt");
    //dataB.Save_File("C:\\MatrixB.txt");
    //inv.Save_File("C:\\Inv.txt");
    //result.Save_File("C:\\result.txt");


    for (i=0 ; i< degree+ 1 ; i++)
    {
        coeff[i] = result.data[0][i];
    }

	
}
CrossRemoval::CrossRemoval(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags)
{
    ui.setupUi(this);

    m_iWidth = 2304;
    m_iHeight = 3200;


    m_pImageYKSrc = new YK16GrayImage(2304,3200);
    m_pImageYKCur = new YK16GrayImage(2304, 3200);
    m_pMedianYKImg = new YK16GrayImage(2304, 3200);

    //m_pMaskHor = new YK16GrayImage(2304,3200); // crosshair = 1 , valid pixels = 0
    //m_pMaskVert= new YK16GrayImage(2304,3200); // crosshair = 1 , valid pixels = 0
    m_pMaskComposite= new YK16GrayImage(m_iWidth,m_iHeight); // crosshair = 1 , valid pixels = 0 //init with 0
    m_pPixelMedianImage= new YK16GrayImage(m_iWidth,m_iHeight); // crosshair = 1 , valid pixels = 0 //init with 0

    //m_pRefImage= new YK16GrayImage(m_iWidth,m_iHeight);

    m_itkCurImage = UnsignedShortImageType::New();
    UnsignedShortImageType::IndexType start;
    start.Fill(0); //basic offset for iteration --> should be 0 if full size image should be processed

    UnsignedShortImageType::SizeType size;
    size[0] = m_iWidth;
    size[1] = m_iHeight;

    UnsignedShortImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    m_itkCurImage->SetRegions(region);
    m_itkCurImage->Allocate();

    m_iCrosshairMargin = DEFAULT_CROSSHAIR_MARGIN;//default = 20
    ui.EditAddMargin->setText(QString("%1").arg(m_iCrosshairMargin));

    m_fSearchROI_X = 0.0; //if this value = 50, only cross-hair candidates in half region of the full image will be recognized as valid.
    m_fSearchROI_Y = 0.0;

	
    ui.EditInclDiffThreshold->setText(QString("%1").arg(INCL_DIFF_THRESHOLD));


	
    m_iFileCnt = 0;	
    m_arrYKImage = NULL;	
    m_arrYKImageMask = NULL;
    m_pCrosshairInfo = NULL;
    m_arrYKImageReplaced=  NULL;

    //m_pRefImage = NULL;	
    m_iRefImageIdx = 0;

    ui.comboBoxSelectView->addItem("0: Original");
    ui.comboBoxSelectView->addItem("1: Replaced");
    ui.comboBoxSelectView->addItem("2: Mask");
}

CrossRemoval::~CrossRemoval()
{
	delete m_pImageYKCur;
	delete m_pImageYKSrc;
	delete m_pMedianYKImg;

	//delete m_pMaskHor;
	//delete m_pMaskVert;
	delete m_pMaskComposite;
	
	delete m_pPixelMedianImage;
	ReleaseMemory(); //delete array 
}

void CrossRemoval::SLT_LoadImage() //MULTIPLE
{
    m_strListSrcFilePath.clear();

    m_strListSrcFilePath = QFileDialog::getOpenFileNames(this,"Select one or more files to open",
        "/home","Images (*.raw)");

    m_iFileCnt = m_strListSrcFilePath.size();

    if (m_iFileCnt < 1)
        return;

    if (m_iFileCnt > 5)
    {
        printf("maximum file cnt =5\n");
        return;
    }

    ReleaseMemory();
    ui.lineEditFileName0->clear();
    ui.lineEditFileName1->clear();
    ui.lineEditFileName2->clear();
    ui.lineEditFileName3->clear();
    ui.lineEditFileName4->clear();

    ui.lineEditNorm0->clear();
    ui.lineEditNorm1->clear();
    ui.lineEditNorm2->clear();
    ui.lineEditNorm3->clear();
    ui.lineEditNorm4->clear();

    ui.radioButtonRef0->setChecked(true);

    m_arrYKImage = new YK16GrayImage [m_iFileCnt];
    m_arrYKImageMask = new YK16GrayImage [m_iFileCnt];
    m_arrYKImageReplaced = new YK16GrayImage [m_iFileCnt];
    m_pCrosshairInfo = new CROSSHAIRINFO [m_iFileCnt];	

    for (int i = 0 ; i < m_iFileCnt ; i++)
    {
        m_arrYKImage[i].CreateImage(m_iWidth, m_iHeight, 0);
        m_arrYKImageMask[i].CreateImage(m_iWidth, m_iHeight, MASK_NONCROSSHAIR);
        m_arrYKImageReplaced[i].CreateImage(m_iWidth, m_iHeight, 0);
    }
    for (int i = 0; i < m_iFileCnt; ++i)
    {		
        QString currentPath = m_strListSrcFilePath.at(i);		
        m_arrYKImage[i].LoadRawImage(currentPath.toLocal8Bit().constData(),m_iWidth,m_iHeight);
    }	

    for (int i= 0 ; i<m_iFileCnt ; i++)
    {
        QFileInfo tmpInfo = QFileInfo(m_strListSrcFilePath.at(i));
		
        if (i == 0)
            ui.lineEditFileName0->setText(tmpInfo.fileName());
        else if (i == 1)
            ui.lineEditFileName1->setText(tmpInfo.fileName());
        else if (i == 2)
            ui.lineEditFileName2->setText(tmpInfo.fileName());
        else if (i == 3)
            ui.lineEditFileName3->setText(tmpInfo.fileName());
        else if (i == 4)
            ui.lineEditFileName4->setText(tmpInfo.fileName());
    }
    m_arrYKImage[0].CalcImageInfo(); //this is reference image

    ui.sliderMin->setValue((int)(m_arrYKImage[0].m_fPixelMean - 5*m_arrYKImage[0].m_fPixelSD));
    ui.sliderMax->setValue((int)(m_arrYKImage[0].m_fPixelMean + 5*m_arrYKImage[0].m_fPixelSD));
    SLT_DrawCurImage(); //Change FileName as w

    /*Original CODE*/

    //1) Open Raw File
    /*QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Raw image file (*.raw)", 0,0);		

      if (!m_pImageYKSrc->LoadRawImage(fileName.toLocal8Bit().constData(),2304,3200))
      return;

      m_strSrcFilePath = fileName;
      m_pImageYKCur->CopyFromBuffer(m_pImageYKSrc->m_pData, m_pImageYKSrc->m_iWidth, m_pImageYKSrc->m_iHeight);

      double mean;
      double SD;
      double max;
      double min;
      m_pImageYKCur->CalcImageInfo(mean, SD, max, min);

      ui.sliderMin->setValue((int)(mean - 4*SD));
      ui.sliderMax->setValue((int)(mean + 4*SD));

      SLT_DrawCurImage();	*/
}

void CrossRemoval::SLT_DrawCurImage()
{
    if (m_iFileCnt == 0)
        return;	

    int winMax = ui.sliderMax->value();
    int winMin = ui.sliderMin->value();

    //labelImage1	
    //labelImage2
    //labelImage3
    //labelImage4
    //labelImage5

    /*int winMaxMask = 1;
      int winMinMask = 0;*/
    /*ui.comboBoxSelectView->addItem("0: Original");
      ui.comboBoxSelectView->addItem("1: Replaced");
      ui.comboBoxSelectView->addItem("2: Mask");*/
    int curIdx = ui.comboBoxSelectView->currentIndex();
		
    if (ui.comboBoxSelectView->currentIndex() == 2)
    {
        for (int i = 0 ; i<m_iFileCnt ; i++)
        {
            m_arrYKImageMask[i].FillPixMapMinMax(0,4);
        }
        switch (m_iFileCnt )
        {	
        case 1:
            ui.labelImage1->SetBaseImage(&m_arrYKImageMask[0]);
            break;
        case 2:
            ui.labelImage1->SetBaseImage(&m_arrYKImageMask[0]);
            ui.labelImage2->SetBaseImage(&m_arrYKImageMask[1]);		

            break;
        case 3:
            ui.labelImage1->SetBaseImage(&m_arrYKImageMask[0]);
            ui.labelImage2->SetBaseImage(&m_arrYKImageMask[1]);
            ui.labelImage3->SetBaseImage(&m_arrYKImageMask[2]);
            break;
        case 4:
            ui.labelImage1->SetBaseImage(&m_arrYKImageMask[0]);
            ui.labelImage2->SetBaseImage(&m_arrYKImageMask[1]);
            ui.labelImage3->SetBaseImage(&m_arrYKImageMask[2]);
            ui.labelImage4->SetBaseImage(&m_arrYKImageMask[3]);
            break;
        case 5:
            ui.labelImage1->SetBaseImage(&m_arrYKImageMask[0]);
            ui.labelImage2->SetBaseImage(&m_arrYKImageMask[1]);
            ui.labelImage3->SetBaseImage(&m_arrYKImageMask[2]);
            ui.labelImage4->SetBaseImage(&m_arrYKImageMask[3]);
            ui.labelImage5->SetBaseImage(&m_arrYKImageMask[4]);
            break;
        default:
            return;
            break;
        }
    }
    else if (ui.comboBoxSelectView->currentIndex() == 1) //replaced
    {
        for (int i = 0 ; i<m_iFileCnt ; i++)
        {
            m_arrYKImageReplaced[i].FillPixMapMinMax(winMin,winMax);
        }

        switch (m_iFileCnt )
        {	
        case 1:
            ui.labelImage1->SetBaseImage(&m_arrYKImageReplaced[0]);
            break;
        case 2:
            ui.labelImage1->SetBaseImage(&m_arrYKImageReplaced[0]);
            ui.labelImage2->SetBaseImage(&m_arrYKImageReplaced[1]);		

            break;
        case 3:
            ui.labelImage1->SetBaseImage(&m_arrYKImageReplaced[0]);
            ui.labelImage2->SetBaseImage(&m_arrYKImageReplaced[1]);
            ui.labelImage3->SetBaseImage(&m_arrYKImageReplaced[2]);
            break;
        case 4:
            ui.labelImage1->SetBaseImage(&m_arrYKImageReplaced[0]);
            ui.labelImage2->SetBaseImage(&m_arrYKImageReplaced[1]);
            ui.labelImage3->SetBaseImage(&m_arrYKImageReplaced[2]);
            ui.labelImage4->SetBaseImage(&m_arrYKImageReplaced[3]);
            break;
        case 5:
            ui.labelImage1->SetBaseImage(&m_arrYKImageReplaced[0]);
            ui.labelImage2->SetBaseImage(&m_arrYKImageReplaced[1]);
            ui.labelImage3->SetBaseImage(&m_arrYKImageReplaced[2]);
            ui.labelImage4->SetBaseImage(&m_arrYKImageReplaced[3]);
            ui.labelImage5->SetBaseImage(&m_arrYKImageReplaced[4]);
            break;
        default:
            return;
            break;
        }


    }
    else if (ui.comboBoxSelectView->currentIndex() == 0) //Original
    {
        for (int i = 0 ; i<m_iFileCnt ; i++)
        {
            m_arrYKImage[i].FillPixMapMinMax(winMin,winMax);
        }

        switch (m_iFileCnt )
        {	
        case 1:
            ui.labelImage1->SetBaseImage(&m_arrYKImage[0]);
            break;
        case 2:
            ui.labelImage1->SetBaseImage(&m_arrYKImage[0]);
            ui.labelImage2->SetBaseImage(&m_arrYKImage[1]);		

            break;
        case 3:
            ui.labelImage1->SetBaseImage(&m_arrYKImage[0]);
            ui.labelImage2->SetBaseImage(&m_arrYKImage[1]);
            ui.labelImage3->SetBaseImage(&m_arrYKImage[2]);
            break;
        case 4:
            ui.labelImage1->SetBaseImage(&m_arrYKImage[0]);
            ui.labelImage2->SetBaseImage(&m_arrYKImage[1]);
            ui.labelImage3->SetBaseImage(&m_arrYKImage[2]);
            ui.labelImage4->SetBaseImage(&m_arrYKImage[3]);
            break;
        case 5:
            ui.labelImage1->SetBaseImage(&m_arrYKImage[0]);
            ui.labelImage2->SetBaseImage(&m_arrYKImage[1]);
            ui.labelImage3->SetBaseImage(&m_arrYKImage[2]);
            ui.labelImage4->SetBaseImage(&m_arrYKImage[3]);
            ui.labelImage5->SetBaseImage(&m_arrYKImage[4]);
            break;
        default:
            return;
            break;
        }
    }

    m_pMaskComposite->FillPixMapMinMax(0,4);
    ui.labelImage_Mask->SetBaseImage(m_pMaskComposite);

    ui.labelImage1->update();
    ui.labelImage2->update();
    ui.labelImage3->update();
    ui.labelImage4->update();
    ui.labelImage5->update();
    ui.labelImage_Mask->update();

    //m_pImageYKCur->DrawToLabel(ui.labelImage);
}

void CrossRemoval::SLT_GoToOriginal()
{
    m_pImageYKCur->CopyFromBuffer(m_pImageYKSrc->m_pData, m_pImageYKSrc->m_iWidth, m_pImageYKSrc->m_iHeight);
    SLT_DrawCurImage();
}

void CrossRemoval::SLT_Median() //Median Filtering
{

    //this->MedianFiltering(m_pImageYKCur, 2);

    CopyYKImage2ItkImage(m_pImageYKCur, m_itkCurImage);

    MedianFilterType::Pointer medianFilter = MedianFilterType::New();
    //MedianFilterType::InputSizeType radius;
    //radius.Fill(2);
	
    medianFilter->SetInput(m_itkCurImage);
    //medianFilter->SetRadius(radius);
    medianFilter->SetRadius(2);

    medianFilter->Update();

    m_itkCurImage = medianFilter->GetOutput();

    //YK16GrayImage::CopyItkImage2YKImage(m_itkCurImage,m_pImageYKCur);
    CopyItkImage2YKImage(m_itkCurImage,m_pMedianYKImg);
    //m_pMedianYKImg->SaveDataAsRaw("C:\\TestMedian.raw");

    m_pImageYKCur->CopyFromBuffer(m_pMedianYKImg->m_pData, m_iWidth , m_iHeight);

    SLT_DrawCurImage();
}

void CrossRemoval::SLT_Gaussian()
{
    CopyYKImage2ItkImage(m_pImageYKCur, m_itkCurImage);

    SmoothingFilterType::Pointer gaussianFilter = SmoothingFilterType::New();
    double sigma = 4.0;	

    gaussianFilter->SetInput(m_itkCurImage);
    gaussianFilter->SetSigma(sigma); //filter specific setting
    gaussianFilter->Update();

    m_itkCurImage = gaussianFilter->GetOutput();

    CopyItkImage2YKImage(m_itkCurImage,m_pImageYKCur);

    SLT_DrawCurImage();
}

void CrossRemoval::SLT_DerivativeHor()
{
    CopyYKImage2ItkImage(m_pImageYKCur, m_itkCurImage);

    DerivativeFilterType::Pointer derivativeFilter = DerivativeFilterType::New();

    derivativeFilter->SetInput(m_itkCurImage);
    derivativeFilter->SetDirection(0); //x axis
    derivativeFilter->Update(); //missed in test code!!!!

    FloatImageType::Pointer itkFloatImage = derivativeFilter->GetOutput(); //no allocation required
    //No need of allocation!!!!		

    FloatImageType::RegionType region = itkFloatImage->GetRequestedRegion();	
    FloatImageType::SizeType tmpSize = region.GetSize();

    int width = tmpSize[0];
    int height = tmpSize[1];

    itk::ImageRegionConstIterator<FloatImageType> it(itkFloatImage, region);

    //Ranges -200 ~ 200
    //use absolute value-based thresholding

    //Image1: Simple Thresholding
    //Image2: Maximum_value based binary

    //unsigned short tmpThreshold = ui.spinThreshold->value();

    int i= 0;
    int j = 0;

    double maxVal = -9999;
    double minVal = 9999;

    for (it.GoToBegin() ; !it.IsAtEnd() ; ++it)
    {
        float tmpVal = it.Get();

        if (tmpVal > maxVal)
            maxVal = tmpVal;
        if (tmpVal < minVal)
            minVal = tmpVal;				
    }

    //Magnification & Shifting the currentImage
    //unsigned short shiftValue = fabs(minVal)*10;

    float tmpMargin = fabs(minVal / 3.0);

    i = 0;
    for (it.GoToBegin() ; !it.IsAtEnd() ; ++it)
    {
        float tmpVal = it.Get();
        if (tmpVal < (-1.0*tmpMargin))		
            m_pImageYKCur->m_pData[i] = BLACK_VALUE;
        else if (tmpVal > (tmpMargin))
            m_pImageYKCur->m_pData[i] = WHITE_VALUE;
        else
            m_pImageYKCur->m_pData[i] = NORMAL_VALUE;
        //m_pImageYKCur->m_pData[i] = (unsigned short)(tmpVal*10 + shiftValue);	 //cur image is float...
        i++;
    }	

    SLT_DrawCurImage();
}

void CrossRemoval::SLT_DerivativeVer()
{
    CopyYKImage2ItkImage(m_pImageYKCur, m_itkCurImage);

    DerivativeFilterType::Pointer derivativeFilter = DerivativeFilterType::New();

    derivativeFilter->SetInput(m_itkCurImage);
    derivativeFilter->SetDirection(1); //y axis
    derivativeFilter->Update(); //missed in test code!!!!

    FloatImageType::Pointer itkFloatImage = derivativeFilter->GetOutput(); //no allocation required
    //No need of allocation!!!!		

    FloatImageType::RegionType region = itkFloatImage->GetRequestedRegion();	
    FloatImageType::SizeType tmpSize = region.GetSize();

    int width = tmpSize[0];
    int height = tmpSize[1];

    itk::ImageRegionConstIterator<FloatImageType> it(itkFloatImage, region);

    //Ranges -200 ~ 200
    //use absolute value-based thresholding

    //Image1: Simple Thresholding
    //Image2: Maximum_value based binary

    //unsigned short tmpThreshold = ui.spinThreshold->value();

    int i= 0;
    int j = 0;
	
    double maxVal = -9999;
    double minVal = 9999;

    for (it.GoToBegin() ; !it.IsAtEnd() ; ++it)
    {
        float tmpVal = it.Get();

        if (tmpVal > maxVal)
            maxVal = tmpVal;
        if (tmpVal < minVal)
            minVal = tmpVal;				
    }

    //Magnification & Shifting the currentImage
    //unsigned short shiftValue = fabs(minVal)*10;

    float tmpMargin = fabs(minVal / 3.0);

    i = 0;
    for (it.GoToBegin() ; !it.IsAtEnd() ; ++it)
    {
        float tmpVal = it.Get();
        if (tmpVal < (-1.0*tmpMargin))		
            m_pImageYKCur->m_pData[i] = BLACK_VALUE;

        else if (tmpVal > (tmpMargin))
            m_pImageYKCur->m_pData[i] = WHITE_VALUE;
        else
            m_pImageYKCur->m_pData[i] = NORMAL_VALUE;
        //m_pImageYKCur->m_pData[i] = (unsigned short)(tmpVal*10 + shiftValue);	 //cur image is float...
        i++;
    }	

    SLT_DrawCurImage();
}

void CrossRemoval::SLT_RemoveCrosshairHor()//Vert --> horizontal line will be removed
{
    QString strAddMargin = ui.EditAddMargin->text();
    m_iCrosshairMargin = strAddMargin.toUInt();

    //Find cross hair point 

    m_fSearchROI_X = ui.EditROIRatioX->text().toDouble();
    m_fSearchROI_Y = ui.EditROIRatioY->text().toDouble();

    if (m_fSearchROI_X <= 0 || m_fSearchROI_X >=1)
        m_fSearchROI_X =DEFAULT_ROI_RATIO_X;

    if (m_fSearchROI_Y <= 0 || m_fSearchROI_Y >=1)
        m_fSearchROI_Y =DEFAULT_ROI_RATIO_Y;

    int innerWidth = m_iWidth * m_fSearchROI_X;
    int innerHeight = m_iHeight * m_fSearchROI_Y;

    int left = (m_iWidth - innerWidth)/2.0;
    int right = left + innerWidth;
    int top = (m_iHeight - innerHeight)/2.0;
    int bottom = top + innerHeight;

    //QRect searchRect = new QRect(left,top, right, bottom);

    int i = 0;
    int j = 0;

    QPoint pt;
	
    std::vector<QPoint> vPoints;
    //std::vector<int> vThicknessVert;

    double sumThickness = 0.0;
    int iCntThickness = 0;

    //Vertical scan
    int iBlackStart1 = 0;
    int iBlackEnd1 = 0;
    int iWhiteStart1 = 0;
    int iWhiteEnd1 = 0;	

    int iBlackStart2 = 0;
    int iBlackEnd2 = 0;
    int iWhiteStart2 = 0;
    int iWhiteEnd2 = 0;	



    for (j = left ; j < right ; j=j+10)
    {
        int prevVal = NORMAL_VALUE;
        int curVal = NORMAL_VALUE;

        iBlackStart1 = -1;
        iBlackEnd1 = -1;
        iWhiteStart1 = -1;
        iWhiteEnd1 = -1;

        iBlackStart2 = -1;
        iBlackEnd2 = -1;
        iWhiteStart2 = -1;
        iWhiteEnd2 = -1;
        int iDetection = 0;
		
        for (i = top ; i< bottom ; i++)
        {
            curVal = (int)m_pImageYKCur->m_pData[m_iWidth*i + j];

            if (iBlackStart1 < 0 || iBlackEnd1 < 0 || iWhiteStart1 < 0  || iWhiteEnd1 < 0)  //if first found center-pixel
            {
                if (curVal == BLACK_VALUE && prevVal != BLACK_VALUE) //Normal --> black
                {
                    iBlackStart1 = i;
                }
                if (curVal != BLACK_VALUE && prevVal == BLACK_VALUE) //Normal --> black
                {
                    iBlackEnd1 = i;
                }
                if (iBlackStart1 >= 0 && iBlackEnd1 >= 0 )
                {
                    if (curVal == WHITE_VALUE && prevVal != WHITE_VALUE) //Normal --> black
                    {
                        iWhiteStart1 = i;
                    }
                    if (curVal != WHITE_VALUE && prevVal == WHITE_VALUE) //Normal --> black
                    {
                        iWhiteEnd1 = i;
                    }
                }			
            }
            else //if one center pixel already found 
            {
                if (curVal == BLACK_VALUE && prevVal != BLACK_VALUE) //Normal --> black
                {
                    iBlackStart2 = i;
                }
                if (curVal != BLACK_VALUE && prevVal == BLACK_VALUE) //Normal --> black
                {
                    iBlackEnd2 = i;
                }
                if (iBlackStart2 >= 0 && iBlackEnd2 >= 0 )// always black line should come first
                {
                    if (curVal == WHITE_VALUE && prevVal != WHITE_VALUE) //Normal --> black
                    {
                        iWhiteStart2 = i;
                    }
                    if (curVal != WHITE_VALUE && prevVal == WHITE_VALUE) //Normal --> black
                    {
                        iWhiteEnd2 = i;
                    }
                }
            }
            prevVal = curVal;
        }// end of for i --> vertical one scan completed

        if (iBlackStart1 > 0 && iBlackEnd1 > 0 && iWhiteStart1>0 && iWhiteEnd1 >0)
            iDetection = 1;
        if (iBlackStart2 > 0 && iBlackEnd2 > 0 && iWhiteStart2>0 && iWhiteEnd2 >0)
            iDetection = 2;

        //only first found pixel will be taken into accunt in vertical scan
        if (iDetection >= 1)
        {
            int tmpX = j;
            int tmpY = (int)((iBlackStart1 + iWhiteEnd1) / 2.0);
            QPoint pt = QPoint(tmpX, tmpY);
            int thickness = iWhiteEnd1 - iBlackStart1;
            if (thickness < 0)
            {
                printf("XVal = %d, Black Start = %d, White End = %d\n", j, iBlackStart1,iWhiteEnd1);
            }

            int curSize = vPoints.size();

            bool IsValidPt = true;

            if (curSize > 2) //Audit this point by inclination
            {
                QPoint prevprevPt = vPoints.at(curSize-2);
                QPoint prevPt = vPoints.at(curSize-1);

                if (prevPt.x() != prevprevPt.x() && pt.x() != prevPt.x())
                {
                    double prevIncl = (prevPt.y() - prevprevPt.y()) / (double)(prevPt.x() - prevprevPt.x());
                    double curIncl = (pt.y() - prevPt.y()) / (double)(pt.x() - prevPt.x());				

                    // value is too different, it is regarded as not on same line.
                    if (fabs(curIncl - prevIncl) > INCL_DIFF_THRESHOLD)				
                        IsValidPt = false;				
                }				
            }
            //m_pImageYKCur->m_pData[m_iWidth*tmpY + tmpX] = (unsigned short)CENTERLINE_VALUE;

            if (IsValidPt)
            {
                vPoints.push_back(pt);
                sumThickness = sumThickness + thickness;
                iCntThickness++;
            }			
            //vThicknessVert.push_back(thickness);
        }
    }//end of j (x)

    /*ofstream fout;
      fout.open("C:\\DebugVPointsHor.txt");
      printf("test\n");

      vector<QPoint>::iterator it;

      for (it = vPoints.begin() ; it != vPoints.end() ; it++)
      {
      fout << (*it).x() << "	" << (*it).y() << endl;
      }
	
      fout.close();*/

    //Calculate Curve Equation ->Least Square

    double coeff[2] = {0.0,0.0};
    g_GetEq(vPoints, 1, coeff);

    double grad = coeff[0];
    double yCut = coeff[1];
    //QString str = QString("%1, %2").arg(grad).arg(yCut);

    int margin = m_iCrosshairMargin;
    int meanHalfThickness =(int)(sumThickness/(double)iCntThickness * 0.5) + margin;
    //printf("sumThickness = %3.2f, iCntThickness = %d, margin = %3.2f",sumThickness,iCntThickness,margin );
    //UpperCurve : yCutUpper = yCut - meanHalfThickness; //draw later
    //UpperCurve : yCutLower = yCut + meanHalfThickness; //draw later

    //int yVal = grad*j + yCut;	

    m_pImageYKCur->CopyFromBuffer(m_pImageYKSrc->m_pData,m_iWidth, m_iHeight);

    //printf("grad = %3.2f, yCut = %3.2f \n", grad, yCut);

    for (j = 0 ; j < m_iWidth ; j++)
    {
        int iStartPt =  grad*j + yCut - meanHalfThickness;
        int iEndPt = grad*j + yCut + meanHalfThickness;
        int tmpVal = 0;

        //printf("iStartPt = %d, iEndPt = %d MeanHalfThickness = %3.2f\n", iStartPt, iEndPt,meanHalfThickness);

        for (i = iStartPt ;  i < iEndPt ; i++)
        {
            if (i < 0 || i >=m_iHeight || j < 0 || j>=m_iWidth)
            {
                //printf("Skip! m_iHeight = %d, m_iWidth = %d , i = %d, j = %d \n", m_iHeight, m_iWidth, i, j);
                continue;
            }

            //printf("grad = %3.2f, yCut = %3.2f \n", grad, yCut);
            tmpVal = GetSubstituteVert(j,i, coeff[0], coeff[1], meanHalfThickness, DEFAULT_SAMPLING_PIXELS, m_pMedianYKImg->m_pData, m_iWidth, m_iHeight);
            //tmpVal = GetSubstituteVert(j,i, coeff[0], coeff[1], meanHalfThickness, DEFAULT_SAMPLING_PIXELS, m_pImageYKSrc->m_pData, m_iWidth, m_iHeight);
            //tmpVal = GetSubstituteVert(j,i, coeff[0], coeff[1], meanHalfThickness, 3, m_pImageYKSrc->m_pData, m_iWidth, m_iHeight);

            //if (tmpVal != 65535) //not dummy value && content with boundary condition
            //	m_pImageYKCur->m_pData[m_iWidth*i+j] = (unsigned short)tmpVal;
            //m_pImageYKCur->m_pData[m_iWidth*i+j] = 0;

            if (tmpVal != 65535) //not dummy value
            {
                m_pImageYKCur->m_pData[m_iWidth*i+j] = (unsigned short)tmpVal;			
            }				
        }		
    }	
    SLT_DrawCurImage();
}

//This function is called after Median --> Gaussian --> X derivative filter 

void CrossRemoval::SLT_RemoveCrosshairVert()//Vert --> vertical line will be removed
{
    QString strAddMargin = ui.EditAddMargin->text();
    m_iCrosshairMargin = strAddMargin.toUInt();

    m_fSearchROI_X = ui.EditROIRatioX->text().toDouble();
    m_fSearchROI_Y = ui.EditROIRatioY->text().toDouble();

    if (m_fSearchROI_X <= 0 || m_fSearchROI_X >=1)
        m_fSearchROI_X =DEFAULT_ROI_RATIO_X;

    if (m_fSearchROI_Y <= 0 || m_fSearchROI_Y >=1)
        m_fSearchROI_Y =DEFAULT_ROI_RATIO_Y;

    //Find cross hair point 
    int innerWidth = m_iWidth * m_fSearchROI_X;
    int innerHeight = m_iHeight * m_fSearchROI_Y;
    int left = (m_iWidth - innerWidth)/2.0;
    int right = left + innerWidth;
    int top = (m_iHeight - innerHeight)/2.0;
    int bottom = top + innerHeight;

    //QRect searchRect = new QRect(left,top, right, bottom);

    int i = 0;
    int j = 0;

    QPoint pt;

    std::vector<QPoint> vPoints;
    //std::vector<int> vThicknessVert;

    double sumThickness = 0.0;
    int iCntThickness = 0;

    //Vertical scan
    int iBlackStart1 = 0;
    int iBlackEnd1 = 0;
    int iWhiteStart1 = 0;
    int iWhiteEnd1 = 0;

    int iBlackStart2 = 0;
    int iBlackEnd2 = 0;
    int iWhiteStart2 = 0;
    int iWhiteEnd2 = 0;

    //for (j = left ; j < right ; j=j+10)
    for (i = top ; i < bottom ; i=i+10)	
    {
        int prevVal = NORMAL_VALUE;
        int curVal = NORMAL_VALUE;

        iBlackStart1 = -1;
        iBlackEnd1 = -1;
        iWhiteStart1 = -1;
        iWhiteEnd1 = -1;

        iBlackStart2 = -1;
        iBlackEnd2 = -1;
        iWhiteStart2 = -1;
        iWhiteEnd2 = -1;

        int iDetection = 0;

        for (j = left ; j< right ; j++)
        {
            curVal = (int)m_pImageYKCur->m_pData[m_iWidth*i + j];

            if (iBlackStart1 < 0 || iBlackEnd1 < 0 || iWhiteStart1 < 0  || iWhiteEnd1 < 0)  //if first found center-pixel
            {
                if (curVal == BLACK_VALUE && prevVal != BLACK_VALUE) //Normal --> black
                {
                    iBlackStart1 = j;
                }
                if (curVal != BLACK_VALUE && prevVal == BLACK_VALUE) //Normal --> black
                {
                    iBlackEnd1 = j;
                }
                if (curVal == WHITE_VALUE && prevVal != WHITE_VALUE) //Normal --> black
                {
                    iWhiteStart1 = j;
                }
                if (curVal != WHITE_VALUE && prevVal == WHITE_VALUE) //Normal --> black
                {
                    iWhiteEnd1 = j;
                }
            }
            else //if one center pixel already found 
            {
                if (curVal == BLACK_VALUE && prevVal != BLACK_VALUE) //Normal --> black
                {
                    iBlackStart2 = j;
                }
                if (curVal != BLACK_VALUE && prevVal == BLACK_VALUE) //Normal --> black
                {
                    iBlackEnd2 = j;
                }
                if (curVal == WHITE_VALUE && prevVal != WHITE_VALUE) //Normal --> black
                {
                    iWhiteStart2 = j;
                }
                if (curVal != WHITE_VALUE && prevVal == WHITE_VALUE) //Normal --> black
                {
                    iWhiteEnd2 = j;
                }
            }			
            prevVal = curVal;
        }

        if (iBlackStart1 > 0 && iBlackEnd1 > 0 && iWhiteStart1>0 && iWhiteEnd1 >0)
            iDetection = 1;
        if (iBlackStart2 > 0 && iBlackEnd2 > 0 && iWhiteStart2>0 && iWhiteEnd2 >0)
            iDetection = 2;


        if (iDetection >= 1)
        {
            int tmpX = (int)((iBlackStart1 + iWhiteEnd1) / 2.0);
            int tmpY = i;
            //QPoint pt = QPoint(tmpX, tmpY);
            QPoint pt = QPoint(tmpY,tmpX); //Fake! to make the line not infinite inclination, X, Y coordinate is changing
            int thickness = iWhiteEnd1 - iBlackStart1;

            //m_pImageYKCur->m_pData[m_iWidth*tmpY + tmpX] = (unsigned short)CENTERLINE_VALUE; //Dotted line 

            int curSize = vPoints.size();

            bool IsValidPt = true;

            if (curSize > 2) //Audit this point by inclination
            {
                QPoint prevprevPt = vPoints.at(curSize-2);
                QPoint prevPt = vPoints.at(curSize-1);

                if (prevPt.x() != prevprevPt.x() && pt.x() != prevPt.x())
                {					
                    double prevIncl = (prevPt.y() - prevprevPt.y()) / (double)(prevPt.x() - prevprevPt.x());
                    double curIncl = (pt.y() - prevPt.y()) / (double)(pt.x() - prevPt.x());				

                    // value is too different, it is regarded as not on same line.
                    double thre = ui.EditInclDiffThreshold->text().toDouble();
                    if (fabs(curIncl - prevIncl) > thre)				
                        IsValidPt = false;				
                }			
            }

            if (IsValidPt)
            {
                vPoints.push_back(pt);
                sumThickness = sumThickness + thickness;
                iCntThickness++;	
            }
        }		
    }
    //Calculate Curve Equation ->Least Square

    int tmpSize = vPoints.size();

    double coeff[2] = {0.0,0.0};
    g_GetEq(vPoints, 1, coeff);

    //ofstream fout;

    //fout.open("C:\\Debug.txt");

    //vector<QPoint>::iterator it;

    //for (it = vPoints.begin() ; it != vPoints.end() ; it++)
    //{
    //	fout << (*it).x() << "	" <<(*it).y() << endl;
    //}

    //fout.close();	

    double grad = coeff[0];//not x - y coordinate but y - x coordinate
    double xCut = coeff[1];//original x value

    //QString str = QString("%1, %2").arg(grad).arg(yCut);

    //int margin = m_iCrosshairMargin;
    int meanHalfThickness =(int)(sumThickness/(double)iCntThickness * 0.5);
    //UpperCurve : yCutUpper = yCut - meanHalfThickness; //draw later
    //UpperCurve : yCutLower = yCut + meanHalfThickness; //draw later

    //int yVal = grad*j + yCut;

    m_pImageYKCur->CopyFromBuffer(m_pImageYKSrc->m_pData,m_iWidth, m_iHeight); //Back to original

    for (i = 0 ; i < m_iHeight ; i++)
    {
        int iStartPt =  grad*i + xCut - meanHalfThickness;
        int iEndPt = grad*i + xCut + meanHalfThickness;
        int tmpVal = 0;

        for (j = iStartPt ;  j < iEndPt ; j++)
        {
            if (i < 0 || i >=m_iHeight || j < 0 || j>=m_iWidth)
                continue;

            tmpVal = GetSubstituteHor(j,i, coeff[0], coeff[1], meanHalfThickness, DEFAULT_SAMPLING_PIXELS, m_pMedianYKImg->m_pData, m_iWidth, m_iHeight);
            //tmpVal = GetSubstituteHor(j,i, coeff[0], coeff[1], meanHalfThickness, DEFAULT_SAMPLING_PIXELS, m_pImageYKSrc->m_pData, m_iWidth, m_iHeight);


            //tmpVal = GetSubstituteVert(j,i, coeff[0], coeff[1], meanHalfThickness, 3, m_pImageYKSrc->m_pData, m_iWidth, m_iHeight);

            if (tmpVal != 65535) //not dummy value
            {
                m_pImageYKCur->m_pData[m_iWidth*i+j] = (unsigned short)tmpVal;
                //m_pMaskVert->m_pData[m_iWidth*i+j] = MASK_CROSSHAIR; // 1 //background --> inherently 0
            }
            //else
            //{
            //	m_pMaskVert->m_pData[m_iWidth*i+j] = MASK_NONCROSSHAIR; // 1 //background --> inherently 0
            //}			
        }
    }	
    SLT_DrawCurImage();
}


unsigned short CrossRemoval::GetSubstituteVert(int fixedX,int y, double coeff0, double coeff1, int halfThickness, int iSamplePixels, unsigned short* medianImg, int width, int height)
{
    double upperMean = 0;
    double lowerMean = 0;
    int i = 0 ; 
    //int j = 0;

    int CenterY = qRound(coeff0 * fixedX + coeff1);

    //if (CenterY >= m_iHeight || CenterY < 0)
    //	return 65535;

    int minIdx = width*(CenterY-halfThickness-iSamplePixels) + fixedX;
    int maxIdx = width*(CenterY+halfThickness+iSamplePixels) + fixedX;

    int size = width * height;	
	
    //CenterY can be larger than max. Height e.g)>= 3200

    if (y < CenterY - halfThickness || y > CenterY + halfThickness)
        return 65535; //no change

    //if (minIdx < 0  || maxIdx >= size)
    //	return 65535; //no calculation

    double tmpSumUpper = 0.0;
    double tmpSumLower = 0.0;
    int cnt = 0;	

    if (minIdx >= 0)
    {
        for (i = CenterY-halfThickness-iSamplePixels ; i < CenterY-halfThickness ; i++)
        {
            tmpSumUpper = tmpSumUpper + medianImg[width*i + fixedX];
            cnt++;
        }
        upperMean = tmpSumUpper / (double)cnt;
    }

    cnt = 0;

    if (maxIdx < size)
    {
        for (i = CenterY+halfThickness ; i < CenterY+halfThickness+iSamplePixels ; i++)
        {
            tmpSumLower = tmpSumLower + medianImg[width*i + fixedX];
            cnt++;
        }
        lowerMean = tmpSumLower / (double)cnt;
    }

    double tmpResult = 0.0;

    if (upperMean > 0 && lowerMean > 0)
    {
        //if difference is too much, choose greater one (to avoid the other cross-hair line's interference
        double maxVal = max(upperMean, lowerMean);
        double minVal = min(upperMean, lowerMean);

        double ratio = maxVal / minVal;
        if (ratio > (100+DEFAULT_PIX_DIFF_PERCENT)/100.0)
            tmpResult = maxVal;
        else
            tmpResult = upperMean + (lowerMean - upperMean) / (double)(2*halfThickness) * (y - (CenterY - halfThickness));
    }
    else if (upperMean > 0 && lowerMean == 0)
        tmpResult = upperMean;
    else if (lowerMean > 0 && upperMean == 0)
        tmpResult = lowerMean;
    else
        tmpResult = 65535; //Dummy value -->remain that pixel as it is

    return (unsigned short)	tmpResult;
}


unsigned short CrossRemoval::GetSubstituteHor(int x,int fixedY, double coeff0, double coeff1, int halfThickness, int iSamplePixels, unsigned short* medianImg, int width, int height)
{
    double leftMean = 0;
    double rightMean = 0;
    //int i = 0 ; 
    int j = 0;

    int CenterX = qRound(coeff0 * fixedY + coeff1);

    int minIdx = width*fixedY + (CenterX - halfThickness - iSamplePixels);		
    int maxIdx = width*fixedY + (CenterX + halfThickness + iSamplePixels);

    int size = width * height;

    if (x < CenterX - halfThickness || x > CenterX + halfThickness)
        return 65535; //no change

    //if (minIdx < 0  || maxIdx >= size)
    //	return 65535; //no calculation

    double tmpSumLeft = 0.0;
    double tmpSumRight = 0.0;
    int cnt = 0;

    if (minIdx >= 0)
    {
        for (j = CenterX-halfThickness-iSamplePixels ; j < CenterX-halfThickness ; j++)
        {
            tmpSumLeft = tmpSumLeft + medianImg[width*fixedY + j];
            cnt++;
        }
        leftMean = tmpSumLeft / (double)cnt;
    }

    cnt = 0;

    if (maxIdx < size)
    {
        for (j = CenterX+halfThickness ; j < CenterX+halfThickness+iSamplePixels ; j++)
        {
            tmpSumRight = tmpSumRight + medianImg[width*fixedY + j];
            cnt++;
        }
        rightMean = tmpSumRight / (double)cnt;
    }
    double tmpResult = 0.0;	

    if (leftMean > 0 && rightMean > 0)
    {
        double maxVal = max(leftMean, rightMean);
        double minVal = min(leftMean, rightMean);

        double ratio = maxVal / minVal;
        if (ratio > (100+DEFAULT_PIX_DIFF_PERCENT)/100.0)
            tmpResult = maxVal;
        else
            tmpResult = leftMean + (rightMean - leftMean) / (double)(2*halfThickness) * (x - (CenterX - halfThickness));
    }
    else if (leftMean > 0 && rightMean == 0)
        tmpResult = leftMean;
    else if (rightMean > 0 && leftMean == 0)
        tmpResult = rightMean;
    else
        tmpResult = 65535; //Dummy value -->remain that pixel as it is	

    return (unsigned short)	tmpResult;
}

void CrossRemoval::SLT_SaveAs()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save Image", "", "Raw image file (*.raw)",0,0);
    m_pImageYKCur->SaveDataAsRaw(fileName.toLocal8Bit().constData());
}

void CrossRemoval::GetCrosshairMask(IMAGEPROC_PARAM& imgProcParam, CROSSHAIRINFO* crossInfo, YK16GrayImage* srcImg, YK16GrayImage* pTargetReplacedImg, YK16GrayImage* pTargetMaskImg, int ReplacementOption)
{
    if (srcImg->IsEmpty())
        return;
    if (pTargetMaskImg->IsEmpty())
        return;

    int medianSize = imgProcParam.MEDIAN_size;
    int gaussianSigma = imgProcParam.GAUSSIAN_sigma;
    double ROI_RatioX = imgProcParam.ROI_RatioX;
    double ROI_RatioY = imgProcParam.ROI_RatioY;
    double continuityThreshold = imgProcParam.continuityThreshold;

    int additionalMargin = imgProcParam.additionalMargin;

    YK16GrayImage* tmpSrcOriginal = new YK16GrayImage(srcImg->m_iWidth, srcImg->m_iHeight);

    YK16GrayImage* tmpSrcHor = new YK16GrayImage(srcImg->m_iWidth, srcImg->m_iHeight);
    YK16GrayImage* tmpSrcVer = new YK16GrayImage(srcImg->m_iWidth, srcImg->m_iHeight);

    YK16GrayImage* tmpMaskHor = new YK16GrayImage(srcImg->m_iWidth, srcImg->m_iHeight);
    YK16GrayImage* tmpMaskVer = new YK16GrayImage(srcImg->m_iWidth, srcImg->m_iHeight);

    YK16GrayImage* tmpMedianImgHor = new YK16GrayImage(srcImg->m_iWidth, srcImg->m_iHeight);
    YK16GrayImage* tmpMedianImgVer = new YK16GrayImage(srcImg->m_iWidth, srcImg->m_iHeight);

    //YK16GrayImage* tmpOriginal = new YK16GrayImage(srcImg->m_iWidth, srcImg->m_iHeight);

//	tmpOriginal->CopyFromBuffer(srcImg->m_pData,srcImg->m_iWidth, srcImg->m_iHeight);
    tmpSrcOriginal->CopyFromBuffer(srcImg->m_pData,srcImg->m_iWidth, srcImg->m_iHeight);
    pTargetReplacedImg->CopyFromBuffer(srcImg->m_pData,srcImg->m_iWidth, srcImg->m_iHeight);

    tmpSrcHor->CopyFromBuffer(srcImg->m_pData, srcImg->m_iWidth, srcImg->m_iHeight);
    //tmpSrcVer->CopyFromBuffer(srcImg->m_pData, srcImg->m_iWidth, srcImg->m_iHeight);

    double Grad = 0.0;
    double yCut = 0.0;	
    int thickness = 0;	

    int i,j;


    //1) Hor line	
    MedianFiltering(tmpSrcHor, medianSize);
    tmpMedianImgHor->CopyFromBuffer(tmpSrcHor->m_pData, tmpSrcHor->m_iWidth, tmpSrcHor->m_iHeight); //original image	

    GaussianFiltering(tmpSrcHor, gaussianSigma);
    DerivativeFiltering(tmpSrcHor, VERTICAL);	
    GetLineEqFromDerivativeImg(HORIZONTAL, tmpSrcHor,&Grad, &yCut,&thickness,ROI_RatioX, ROI_RatioY, continuityThreshold); // 0:hor, 1: vert
    //Replace those pixels

    int halfThickness = (int)(thickness/2.0 + additionalMargin);
    for (j = 0 ; j < m_iWidth ; j++)
    {
        int iStartPt =  Grad*j + yCut - halfThickness;
        int iEndPt = Grad*j + yCut + halfThickness;
        int tmpVal = 0;		

        for (i = iStartPt ;  i < iEndPt ; i++)
        {
            if (i < 0 || i >=m_iHeight || j < 0 || j>=m_iWidth)
            {				
                continue;
            }
			
            /*REplacement option */
            if (ReplacementOption == XY_FROM_MEDIAN)
            {
                tmpVal = GetSubstituteVert(j,i, Grad, yCut, halfThickness, DEFAULT_SAMPLING_PIXELS, tmpMedianImgHor->m_pData, tmpMedianImgHor->m_iWidth, tmpMedianImgHor->m_iHeight);
            }
            else if (ReplacementOption == Y_FROM_ORIGINAL) //only when oblique
            {
                tmpVal = GetSubstituteVert(j,i, Grad, yCut, halfThickness, DEFAULT_SAMPLING_PIXELS, tmpSrcOriginal->m_pData, tmpSrcOriginal->m_iWidth, tmpSrcOriginal->m_iHeight);
            }


            if (tmpVal != 65535) //not dummy value
            {
                pTargetReplacedImg->m_pData[m_iWidth*i+j] = (unsigned short)tmpVal;
            }				
        }		
    }
    crossInfo->GradientHor = Grad;
    crossInfo->yCutHor = yCut;
    crossInfo->thickenssHor = thickness;
    GenerateMaskImgForSingle(HORIZONTAL, tmpMaskHor, (int)(thickness/2.0+additionalMargin), Grad, yCut);//margin: half margin	


    //2) Ver line

    tmpSrcOriginal->CopyFromBuffer(pTargetReplacedImg->m_pData, m_iWidth, m_iHeight ); //new original
    tmpSrcVer->CopyFromBuffer(pTargetReplacedImg->m_pData, m_iWidth, m_iHeight );

    MedianFiltering(tmpSrcVer, medianSize);	
    tmpMedianImgVer->CopyFromBuffer(tmpSrcVer->m_pData, tmpSrcVer->m_iWidth, tmpSrcVer->m_iHeight);
    GaussianFiltering(tmpSrcVer, gaussianSigma);
	
    if (ReplacementOption == XY_FROM_MEDIAN)
    {
        DerivativeFiltering(tmpSrcVer, HORIZONTAL);
        GetLineEqFromDerivativeImg(VERTICAL, tmpSrcVer,&Grad, &yCut,&thickness,ROI_RatioX, ROI_RatioY, continuityThreshold); // 0:hor, 1: vert

        halfThickness = (int)(thickness/2.0 + additionalMargin);

        for (i = 0 ; i < m_iHeight ; i++)
        {
            int iStartPt =  Grad*i + yCut - halfThickness;
            int iEndPt = Grad*i + yCut + halfThickness;
            int tmpVal = 0;

            for (j = iStartPt ;  j < iEndPt ; j++)
            {
                if (i < 0 || i >=m_iHeight || j < 0 || j>=m_iWidth)
                    continue;

                tmpVal = GetSubstituteHor(j,i, Grad, yCut, halfThickness, DEFAULT_SAMPLING_PIXELS, tmpMedianImgVer->m_pData, m_iWidth, m_iHeight);	

                if (tmpVal != 65535) //not dummy value
                {
                    pTargetReplacedImg->m_pData[m_iWidth*i+j] = (unsigned short)tmpVal;				
                }			
            }
        }	
        crossInfo->GradientVer = Grad;
        crossInfo->yCutVer = yCut;
        crossInfo->thickenssVer = thickness;
        GenerateMaskImgForSingle(VERTICAL, tmpMaskVer, (int)(thickness/2.0+additionalMargin), Grad, yCut);//margin: half margin
	
    }
    else if (ReplacementOption == Y_FROM_ORIGINAL)
    {
        DerivativeFiltering(tmpSrcVer, VERTICAL);	
        GetLineEqFromDerivativeImg(HORIZONTAL, tmpSrcVer,&Grad, &yCut,&thickness,ROI_RatioX, ROI_RatioY, continuityThreshold); // 0:hor, 1: vert
        //Replace those pixels

        int halfThickness = (int)(thickness/2.0 + additionalMargin);
        for (j = 0 ; j < m_iWidth ; j++)
        {
            int iStartPt =  Grad*j + yCut - halfThickness;
            int iEndPt = Grad*j + yCut + halfThickness;
            int tmpVal = 0;		

            for (i = iStartPt ;  i < iEndPt ; i++)
            {
                if (i < 0 || i >=m_iHeight || j < 0 || j>=m_iWidth)
                {				
                    continue;
                }

                /*REplacement option */
                tmpVal = GetSubstituteVert(j,i, Grad, yCut, halfThickness, DEFAULT_SAMPLING_PIXELS, tmpSrcOriginal->m_pData, tmpSrcOriginal->m_iWidth, tmpSrcOriginal->m_iHeight);			


                if (tmpVal != 65535) //not dummy value
                {
                    pTargetReplacedImg->m_pData[m_iWidth*i+j] = (unsigned short)tmpVal;
                }				
            }		
        }
        crossInfo->GradientVer = Grad;
        crossInfo->yCutVer = yCut;
        crossInfo->thickenssVer = thickness;
        GenerateMaskImgForSingle(HORIZONTAL, tmpMaskVer, (int)(thickness/2.0+additionalMargin), Grad, yCut);//margin: half margin
	
    }

    int size = srcImg->m_iWidth * srcImg->m_iHeight;

    for (int i = 0 ; i<size ; i++)
    {
        if( tmpMaskHor->m_pData[i] == MASK_CROSSHAIR || tmpMaskVer->m_pData[i] == MASK_CROSSHAIR)
            pTargetMaskImg->m_pData[i] = MASK_CROSSHAIR;
        else
            pTargetMaskImg->m_pData[i] = MASK_NONCROSSHAIR;
    }

    crossInfo->pMaskImg = pTargetMaskImg;
	
	
    //pTargetMaskImg->SaveDataAsRaw("C:\\CompositeMaskInside.raw");

    delete tmpSrcHor;
    delete tmpSrcVer;
    delete tmpMaskHor;
    delete tmpMaskVer;

    delete tmpMedianImgHor;
    delete tmpMedianImgVer;

    delete tmpSrcOriginal;
}

void CrossRemoval::SLT_RemoveCrosshairMacro()
{
    /*Original code below */

    if (m_pImageYKSrc->IsEmpty())
        return;
    //Horizontal crosshair first

    QProgressDialog* dlgProgBar = new QProgressDialog("Operation in progress.", "Cancel", 0, 100);
    dlgProgBar->setWindowModality(Qt::WindowModal);

    dlgProgBar->setValue(5);
    dlgProgBar->setLabelText("Removing horizontal line: Median filtering");
    //1)Median Filtering
    SLT_Median(); //tmpImage is saved in m_pMedianYKImg
    dlgProgBar->setValue(20);
    dlgProgBar->setLabelText("Removing horizontal line: Gaussian filtering");
    SLT_Gaussian();
    dlgProgBar->setValue(30);
    dlgProgBar->setLabelText("Removing horizontal line: Derivative filtering");
    SLT_DerivativeVer();
    dlgProgBar->setValue(40);
    dlgProgBar->setLabelText("Removing horizontal line: Replacing crosshair-region with neiborhood pixels");
    SLT_RemoveCrosshairHor(); //will refer to Median image and Cur image which was copied from original src image
    //will generate new curImage only crosshair was removed (no median, no other filter)
    dlgProgBar->setValue(50);
    dlgProgBar->setLabelText("Reset source image: horizontal line-removed image");
    SLT_SetCurImageAsNewSourceImage();

    dlgProgBar->setValue(55);
    dlgProgBar->setLabelText("Removing vertical line: Median filtering");
    SLT_Median(); //tmpImage is saved in m_pMedianYKImg

    dlgProgBar->setValue(70);
    dlgProgBar->setLabelText("Removing vertical line: Gaussian filtering");
    SLT_Gaussian();

    dlgProgBar->setValue(80);
    dlgProgBar->setLabelText("Removing vertical line: Derivative filtering");
    SLT_DerivativeHor();

    dlgProgBar->setValue(90);
    dlgProgBar->setLabelText("Removing vertical line: Replacing crosshair-region with neiborhood pixels");
    SLT_RemoveCrosshairVert(); //will refer to Median image and Cur image which was copied from original src image

    dlgProgBar->setValue(95);	
    dlgProgBar->setLabelText("Cross-hair removal was successful and saving to the file.");

    if (!SaveAutoFileName(m_pImageYKCur, m_strSrcFilePath, "_CR")) //CR means crosshair removed
    {
        printf("error in saving file\n");
    }			

    //Save as
    dlgProgBar->setValue(100);
    delete dlgProgBar;
}

void CrossRemoval::SLT_SetCurImageAsNewSourceImage()
{
    m_pImageYKSrc->CopyFromBuffer(m_pImageYKCur->m_pData,m_iWidth, m_iHeight);
}

bool CrossRemoval::SaveAutoFileName(QString& srcFilePath, QString endFix)
{
    if (srcFilePath.length() < 2)
        return false;

    QFileInfo srcFileInfo = QFileInfo(srcFilePath);

    QDir dir = srcFileInfo.absoluteDir();
    QString baseName = srcFileInfo.baseName();
    QString extName = srcFileInfo.completeSuffix();

    QString newFileName = baseName.append(endFix).append(".").append(extName);
    QString newPath = dir.absolutePath() + "\\" + newFileName;	

    m_pImageYKCur->SaveDataAsRaw(newPath.toLocal8Bit().constData());
//	m_pMaskComposite->SaveDataAsRaw(newPath.toLocal8Bit().constData());

    return true;
}

bool CrossRemoval::SaveAutoFileName( YK16GrayImage* pYK16Img, QString& srcFilePath, QString endFix )
{
    if (srcFilePath.length() < 2)
        return false;

    QFileInfo srcFileInfo = QFileInfo(srcFilePath);

    QDir dir = srcFileInfo.absoluteDir();
    QString baseName = srcFileInfo.baseName();
    QString extName = srcFileInfo.completeSuffix();

    QString newFileName = baseName.append(endFix).append(".").append(extName);
    QString newPath = dir.absolutePath() + "\\" + newFileName;	

    pYK16Img->SaveDataAsRaw(newPath.toLocal8Bit().constData());
    //pMaskComposite->SaveDataAsRaw(newPath.toLocal8Bit().constData());

    return true;

}

bool CrossRemoval::MedianFiltering( YK16GrayImage* pImage, int medWindow )
{
//	return true;//YKTEMP

    if (pImage == NULL)
        return false;

    UnsignedShortImageType::Pointer itkCurImage = UnsignedShortImageType::New();
    UnsignedShortImageType::IndexType start;
    start.Fill(0); //basic offset for iteration --> should be 0 if full size image should be processed

    UnsignedShortImageType::SizeType size;
    size[0] = pImage->m_iWidth;
    size[1] = pImage->m_iHeight;

    UnsignedShortImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    itkCurImage->SetRegions(region);
    itkCurImage->Allocate();

    CopyYKImage2ItkImage(pImage, itkCurImage);

    MedianFilterType::Pointer medianFilter = MedianFilterType::New();

    medianFilter->SetInput(itkCurImage);
    medianFilter->SetRadius(medWindow);
    medianFilter->Update();

    itkCurImage = medianFilter->GetOutput();
    //YK16GrayImage::CopyItkImage2YKImage(m_itkCurImage,m_pImageYKCur);
    CopyItkImage2YKImage(itkCurImage,pImage);
    //m_pMedianYKImg->SaveDataAsRaw("C:\\TestMedian.raw");
    //m_pImageYKCur->CopyFromBuffer(m_pMedianYKImg->m_pData, m_iWidth , m_iHeight);

    //itkCurImage->Delete();

    return true;
}

bool CrossRemoval::GaussianFiltering( YK16GrayImage* pImage, double sigma )
{
    if (pImage == NULL)
        return false;

    UnsignedShortImageType::Pointer itkCurImage = UnsignedShortImageType::New();
    UnsignedShortImageType::IndexType start;
    start.Fill(0); //basic offset for iteration --> should be 0 if full size image should be processed

    UnsignedShortImageType::SizeType size;
    size[0] = pImage->m_iWidth;
    size[1] = pImage->m_iHeight;

    UnsignedShortImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    itkCurImage->SetRegions(region);
    itkCurImage->Allocate();


    CopyYKImage2ItkImage(pImage, itkCurImage);

    SmoothingFilterType::Pointer gaussianFilter = SmoothingFilterType::New();
    //double sigma = sigma;	

    gaussianFilter->SetInput(itkCurImage);
    gaussianFilter->SetSigma(sigma); //filter specific setting
    gaussianFilter->Update();

    itkCurImage = gaussianFilter->GetOutput();

    CopyItkImage2YKImage(itkCurImage,pImage);

    //itkCurImage->Delete();

    //SLT_DrawCurImage();
    return true;
}

bool CrossRemoval::DerivativeFiltering( YK16GrayImage* pImage, int direction )
{
	if (pImage == NULL)
		return false;

	UnsignedShortImageType::Pointer itkCurImage = UnsignedShortImageType::New();
	UnsignedShortImageType::IndexType start;
	start.Fill(0); //basic offset for iteration --> should be 0 if full size image should be processed

	UnsignedShortImageType::SizeType size;
	size[0] = pImage->m_iWidth;
	size[1] = pImage->m_iHeight;

	UnsignedShortImageType::RegionType region;
	region.SetSize(size);
	region.SetIndex(start);

	itkCurImage->SetRegions(region);
	itkCurImage->Allocate();

	CopyYKImage2ItkImage(pImage, itkCurImage);	

	DerivativeFilterType::Pointer derivativeFilter = DerivativeFilterType::New();

	derivativeFilter->SetInput(itkCurImage);
	derivativeFilter->SetDirection(direction); //0 = x axis, 1= y axis
	derivativeFilter->Update(); //missed in test code!!!!

	FloatImageType::Pointer itkFloatImage = derivativeFilter->GetOutput(); //no allocation required
	//No need of allocation!!!!		

	FloatImageType::RegionType regionFloat = itkFloatImage->GetRequestedRegion();	
	FloatImageType::SizeType tmpSize = region.GetSize();

	int width = tmpSize[0];
	int height = tmpSize[1];

	
	int i= 0;
	int j = 0;

	double maxVal = -9999;
	double minVal = 9999;

	itk::ImageRegionConstIterator<FloatImageType> it(itkFloatImage, regionFloat);	

	for (it.GoToBegin() ; !it.IsAtEnd() ; ++it)
	{
		float tmpVal = it.Get();

		if (tmpVal > maxVal)
			maxVal = tmpVal;
		if (tmpVal < minVal)
			minVal = tmpVal;				
	}

	//Magnification & Shifting the currentImage
	//unsigned short shiftValue = fabs(minVal)*10;

	float tmpMargin = fabs(minVal / 3.0);
	i = 0;

	for (it.GoToBegin() ; !it.IsAtEnd() ; ++it)
	{
		float tmpVal = it.Get();
		if (tmpVal < (-1.0*tmpMargin))		
			pImage->m_pData[i] = BLACK_VALUE;
		else if (tmpVal > (tmpMargin))
			pImage->m_pData[i] = WHITE_VALUE;
		else
			pImage->m_pData[i] = NORMAL_VALUE;
		//m_pImageYKCur->m_pData[i] = (unsigned short)(tmpVal*10 + shiftValue);	 //cur image is float...
		i++;
	}

	//YK16GrayImage::CopyItkImage2YKImage(itkCurImage,pImage);
//	derivativeFilter->Delete();
	//itkCurImage->Delete();

	return true;
}
//Source Image: Derivative Image
void CrossRemoval::GetLineEqFromDerivativeImg(int direction, YK16GrayImage* pDerivativeImage, double* Grad, double* yCut, int* thickness, double ROI_RatioX, double ROI_RatioY, double inclThreshold) // 0:hor, 1: vert
{
    //QString strAddMargin = ui.EditAddMargin->text();
    //m_iCrosshairMargin = strAddMargin.toUInt();

    if (ROI_RatioX <= 0 || ROI_RatioX >=1)
        ROI_RatioX =DEFAULT_ROI_RATIO_X;

    if (ROI_RatioY <= 0 || ROI_RatioY >=1)
        ROI_RatioY =DEFAULT_ROI_RATIO_Y;

    int width = pDerivativeImage->m_iWidth;
    int height = pDerivativeImage->m_iHeight;

    int innerWidth = width * ROI_RatioX;
    int innerHeight = height * ROI_RatioY;

    int left = (width - innerWidth)/2.0;
    int right = left + innerWidth;
    int top = (height - innerHeight)/2.0;
    int bottom = top + innerHeight;

    //QRect searchRect = new QRect(left,top, right, bottom);

    int i = 0;
    int j = 0;

    QPoint pt;
	
    std::vector<QPoint> vPoints;
    //std::vector<int> vThicknessVert;

    double sumThickness = 0.0;
    int iCntThickness = 0;

    //Vertical scan
    int iBlackStart1 = 0;
    int iBlackEnd1 = 0;
    int iWhiteStart1 = 0;
    int iWhiteEnd1 = 0;	

    int iBlackStart2 = 0;
    int iBlackEnd2 = 0;
    int iWhiteStart2 = 0;
    int iWhiteEnd2 = 0;	

    if (direction == HORIZONTAL)
    {
        for (j = left ; j < right ; j=j+10)
        {
            int prevVal = NORMAL_VALUE;
            int curVal = NORMAL_VALUE;

            iBlackStart1 = -1;
            iBlackEnd1 = -1;
            iWhiteStart1 = -1;
            iWhiteEnd1 = -1;

            iBlackStart2 = -1;
            iBlackEnd2 = -1;
            iWhiteStart2 = -1;
            iWhiteEnd2 = -1;
            int iDetection = 0;

            for (i = top ; i< bottom ; i++)
            {
                curVal = (int)pDerivativeImage->m_pData[m_iWidth*i + j];

                if (iBlackStart1 < 0 || iBlackEnd1 < 0 || iWhiteStart1 < 0  || iWhiteEnd1 < 0)  //if first found center-pixel
                {
                    if (curVal == BLACK_VALUE && prevVal != BLACK_VALUE) //Normal --> black
                    {
                        iBlackStart1 = i;
                    }
                    if (curVal != BLACK_VALUE && prevVal == BLACK_VALUE) //Normal --> black
                    {
                        iBlackEnd1 = i;
                    }
                    if (iBlackStart1 >= 0 && iBlackEnd1 >= 0 )
                    {
                        if (curVal == WHITE_VALUE && prevVal != WHITE_VALUE) //Normal --> black
                        {
                            iWhiteStart1 = i;
                        }
                        if (curVal != WHITE_VALUE && prevVal == WHITE_VALUE) //Normal --> black
                        {
                            iWhiteEnd1 = i;
                        }
                    }			
                }
                else //if one center pixel already found 
                {
                    if (curVal == BLACK_VALUE && prevVal != BLACK_VALUE) //Normal --> black
                    {
                        iBlackStart2 = i;
                    }
                    if (curVal != BLACK_VALUE && prevVal == BLACK_VALUE) //Normal --> black
                    {
                        iBlackEnd2 = i;
                    }
                    if (iBlackStart2 >= 0 && iBlackEnd2 >= 0 )// always black line should come first
                    {
                        if (curVal == WHITE_VALUE && prevVal != WHITE_VALUE) //Normal --> black
                        {
                            iWhiteStart2 = i;
                        }
                        if (curVal != WHITE_VALUE && prevVal == WHITE_VALUE) //Normal --> black
                        {
                            iWhiteEnd2 = i;
                        }
                    }
                }
                prevVal = curVal;
            }// end of for i --> vertical one scan completed

            if (iBlackStart1 > 0 && iBlackEnd1 > 0 && iWhiteStart1>0 && iWhiteEnd1 >0)
                iDetection = 1;
            if (iBlackStart2 > 0 && iBlackEnd2 > 0 && iWhiteStart2>0 && iWhiteEnd2 >0)
                iDetection = 2;

            //only first found pixel will be taken into accunt in vertical scan
            if (iDetection >= 1)
            {
                int tmpX = j;
                int tmpY = (int)((iBlackStart1 + iWhiteEnd1) / 2.0);
                QPoint pt = QPoint(tmpX, tmpY);
                int thickness = iWhiteEnd1 - iBlackStart1;
                if (thickness < 0)
                {
                    printf("XVal = %d, Black Start = %d, White End = %d\n", j, iBlackStart1,iWhiteEnd1);
                }

                int curSize = vPoints.size();

                bool IsValidPt = true;

                if (curSize > 2) //Audit this point by inclination
                {
                    QPoint prevprevPt = vPoints.at(curSize-2);
                    QPoint prevPt = vPoints.at(curSize-1);

                    if (prevPt.x() != prevprevPt.x() && pt.x() != prevPt.x())
                    {
                        double prevIncl = (prevPt.y() - prevprevPt.y()) / (double)(prevPt.x() - prevprevPt.x());
                        double curIncl = (pt.y() - prevPt.y()) / (double)(pt.x() - prevPt.x());				

                        // value is too different, it is regarded as not on same line.
                        if (fabs(curIncl - prevIncl) > inclThreshold)				
                            IsValidPt = false;				
                    }				
                }
                //m_pImageYKCur->m_pData[m_iWidth*tmpY + tmpX] = (unsigned short)CENTERLINE_VALUE;

                if (IsValidPt)
                {
                    vPoints.push_back(pt);
                    sumThickness = sumThickness + thickness;
                    iCntThickness++;
                }			
                //vThicknessVert.push_back(thickness);
            }
        }//end of j (x)
    }
    else if (direction == VERTICAL)
    {
        for (i = top ; i < bottom ; i=i+10)	
        {
            int prevVal = NORMAL_VALUE;
            int curVal = NORMAL_VALUE;

            iBlackStart1 = -1;
            iBlackEnd1 = -1;
            iWhiteStart1 = -1;
            iWhiteEnd1 = -1;

            iBlackStart2 = -1;
            iBlackEnd2 = -1;
            iWhiteStart2 = -1;
            iWhiteEnd2 = -1;

            int iDetection = 0;

            for (j = left ; j< right ; j++)
            {
                curVal = (int)pDerivativeImage->m_pData[m_iWidth*i + j];

                if (iBlackStart1 < 0 || iBlackEnd1 < 0 || iWhiteStart1 < 0  || iWhiteEnd1 < 0)  //if first found center-pixel
                {
                    if (curVal == BLACK_VALUE && prevVal != BLACK_VALUE) //Normal --> black
                    {
                        iBlackStart1 = j;
                    }
                    if (curVal != BLACK_VALUE && prevVal == BLACK_VALUE) //Normal --> black
                    {
                        iBlackEnd1 = j;
                    }
                    if (curVal == WHITE_VALUE && prevVal != WHITE_VALUE) //Normal --> black
                    {
                        iWhiteStart1 = j;
                    }
                    if (curVal != WHITE_VALUE && prevVal == WHITE_VALUE) //Normal --> black
                    {
                        iWhiteEnd1 = j;
                    }
                }
                else //if one center pixel already found 
                {
                    if (curVal == BLACK_VALUE && prevVal != BLACK_VALUE) //Normal --> black
                    {
                        iBlackStart2 = j;
                    }
                    if (curVal != BLACK_VALUE && prevVal == BLACK_VALUE) //Normal --> black
                    {
                        iBlackEnd2 = j;
                    }
                    if (curVal == WHITE_VALUE && prevVal != WHITE_VALUE) //Normal --> black
                    {
                        iWhiteStart2 = j;
                    }
                    if (curVal != WHITE_VALUE && prevVal == WHITE_VALUE) //Normal --> black
                    {
                        iWhiteEnd2 = j;
                    }
                }			
                prevVal = curVal;
            }

            if (iBlackStart1 > 0 && iBlackEnd1 > 0 && iWhiteStart1>0 && iWhiteEnd1 >0)
                iDetection = 1;
            if (iBlackStart2 > 0 && iBlackEnd2 > 0 && iWhiteStart2>0 && iWhiteEnd2 >0)
                iDetection = 2;


            if (iDetection >= 1)
            {
                int tmpX = (int)((iBlackStart1 + iWhiteEnd1) / 2.0);
                int tmpY = i;
                //QPoint pt = QPoint(tmpX, tmpY);
                QPoint pt = QPoint(tmpY,tmpX); //Fake! to make the line not infinite inclination, X, Y coordinate is changing
                int thickness = iWhiteEnd1 - iBlackStart1;

                //m_pImageYKCur->m_pData[m_iWidth*tmpY + tmpX] = (unsigned short)CENTERLINE_VALUE; //Dotted line 

                int curSize = vPoints.size();

                bool IsValidPt = true;

                if (curSize > 2) //Audit this point by inclination
                {
                    QPoint prevprevPt = vPoints.at(curSize-2);
                    QPoint prevPt = vPoints.at(curSize-1);

                    if (prevPt.x() != prevprevPt.x() && pt.x() != prevPt.x())
                    {					
                        double prevIncl = (prevPt.y() - prevprevPt.y()) / (double)(prevPt.x() - prevprevPt.x());
                        double curIncl = (pt.y() - prevPt.y()) / (double)(pt.x() - prevPt.x());				

                        // value is too different, it is regarded as not on same line.
                        //double thre = ui.EditInclDiffThreshold->text().toDouble();
                        if (fabs(curIncl - prevIncl) > inclThreshold)				
                            IsValidPt = false;				
                    }			
                }

                if (IsValidPt)
                {
                    vPoints.push_back(pt);
                    sumThickness = sumThickness + thickness;
                    iCntThickness++;	
                }
            }		
        }
    }

	
    //Calculate Curve Equation ->Least Square
    double coeff[2] = {0.0,0.0};
    g_GetEq(vPoints, 1, coeff);

    (*Grad) = coeff[0];
    (*yCut) = coeff[1];
    //QString str = QString("%1, %2").arg(grad).arg(yCut);

    //int margin = m_iCrosshairMargin;
    //int meanHalfThickness =(int)(sumThickness/(double)iCntThickness * 0.5) + margin;

    (*thickness) = (int)(sumThickness/(double)iCntThickness);
    //printf("sumThickness = %3.2f, iCntThickness = %d, margin = %3.2f",sumThickness,iCntThickness,margin );
    //UpperCurve : yCutUpper = yCut - meanHalfThickness; //draw later
    //UpperCurve : yCutLower = yCut + meanHalfThickness; //draw later

    //int yVal = grad*j + yCut;
    //m_pImageYKCur->CopyFromBuffer(m_pImageYKSrc->m_pData,m_iWidth, m_iHeight);
}

void CrossRemoval::GenerateMaskImgForSingle( int direction, YK16GrayImage* pTargetMaskImg, int margin, double Grad, double yCut )
{
    if (pTargetMaskImg == NULL)
        return;	

    int width; 
    int height;
    //int size = width*height;

    double yCutUpper = yCut + margin;
    double yCutLower = yCut - margin;
    int i,j;
	
    if (direction == HORIZONTAL) //Normal coordinate
    {
        width = pTargetMaskImg->m_iWidth;
        height = pTargetMaskImg->m_iHeight;	

        for (i = 0 ; i< height ; i++)
        {
            for (j = 0 ; j <width ; j++)
            {
                double yUpper = Grad*j+yCutUpper;
                double yLower = Grad*j+yCutLower;

                if (i <= yUpper && i >=yLower)
                    pTargetMaskImg->m_pData[width*i+j] = MASK_CROSSHAIR;
                else
                    pTargetMaskImg->m_pData[width*i+j] = MASK_NONCROSSHAIR;				
            }
        }
    }
    else if (direction == VERTICAL)
    {
        width = pTargetMaskImg->m_iHeight; //Rotate coordinate
        height = pTargetMaskImg->m_iWidth;

        for (i = 0 ; i< height ; i++)
        {
            for (j = 0 ; j <width ; j++)
            {
                double yUpper = Grad*j+yCutUpper;
                double yLower = Grad*j+yCutLower;

                if (i <= yUpper && i >=yLower)
                    pTargetMaskImg->m_pData[height*j +i] = MASK_CROSSHAIR;
                else
                    pTargetMaskImg->m_pData[height*j +i] = MASK_NONCROSSHAIR;				
            }
        }

    }
}

void CrossRemoval::ReleaseMemory()
{	
    if (m_arrYKImage != NULL)
    {
        delete [] m_arrYKImage;
        m_arrYKImage = NULL;
    }	

    if (m_arrYKImageMask != NULL)
    {
        delete [] m_arrYKImageMask;
        m_arrYKImage = NULL;
    }

    if(m_pCrosshairInfo != NULL)
    {
        delete [] m_pCrosshairInfo;
        m_pCrosshairInfo = NULL;
    }
    if (m_arrYKImageReplaced != NULL)
    {
        delete [] m_arrYKImageReplaced;
        m_arrYKImageReplaced = NULL;
    }

}

void CrossRemoval::SLT_CrosshairDetection() //Normal
{
    IMAGEPROC_PARAM imgProcParam;
    imgProcParam.MEDIAN_size = 2;
    imgProcParam.GAUSSIAN_sigma = 4;
    imgProcParam.continuityThreshold = ui.EditInclDiffThreshold->text().toDouble();
    imgProcParam.ROI_RatioX = ui.EditROIRatioX->text().toDouble();
    imgProcParam.ROI_RatioY = ui.EditROIRatioY->text().toDouble();
    imgProcParam.additionalMargin = ui.EditAddMargin->text().toDouble();

    //CROSSHAIRINFO crosshairInfo;

    for (int i = 0 ; i<m_iFileCnt ; i++)
    {
        if (ui.checkBoxVertReplacement->isChecked())
            GetCrosshairMask(imgProcParam,&m_pCrosshairInfo[i],&m_arrYKImage[i], &m_arrYKImageReplaced[i], &m_arrYKImageMask[i], Y_FROM_ORIGINAL); //pixel replacement also willbe done here
        else
            GetCrosshairMask(imgProcParam,&m_pCrosshairInfo[i],&m_arrYKImage[i], &m_arrYKImageReplaced[i], &m_arrYKImageMask[i], XY_FROM_MEDIAN); //pixel replacement also willbe done here

        m_arrYKImageMask[i].CalcImageInfo();
        printf("Image index = %d: mean = %3.2f, MAX = %3.2f\n", i, m_arrYKImageMask[i].m_fPixelMean, m_arrYKImageMask[i].m_fPixelMax);
    }	
    //Calculate composite mask 	
    int size = m_iWidth*m_iHeight;

    for (int k = 0 ; k < size ; k++)
    {
        bool normalPixExist = false;
        for (int i = 0 ; i<m_iFileCnt ; i++)
        {
            if (m_arrYKImageMask[i].m_pData[k] == MASK_NONCROSSHAIR)
            {
                normalPixExist = true;
                break;
            }
        }
        if (normalPixExist)
        {
            m_pMaskComposite->m_pData[k] = MASK_NONCROSSHAIR;
        }
        else
        {
            m_pMaskComposite->m_pData[k] = MASK_CROSSHAIR;
        }
    }

    //m_pMaskComposite->SaveDataAsRaw("C:\\CompositeCrosshair.raw");

    SLT_DrawCurImage();	
    return;
}

void CrossRemoval::SLT_SetRefImage()
{
    if (ui.radioButtonRef0->isChecked())
    {
        bool test  = false;
        if (m_iFileCnt < 1)
        {
            m_iRefImageIdx = -1;
        }
        else //normal
        {
            m_iRefImageIdx = 0;
        }
    }

    else if (ui.radioButtonRef1->isChecked())
    {
        if (m_iFileCnt < 2)
        {
            m_iRefImageIdx = 0;
            ui.radioButtonRef0->setChecked(true);

        }
        else //normal
            m_iRefImageIdx = 1;
    }

    else if (ui.radioButtonRef2->isChecked())
    {
        if (m_iFileCnt < 3)
        {
            m_iRefImageIdx = 0;
            ui.radioButtonRef0->setChecked(true);			

        }
        else //normal
            m_iRefImageIdx = 2;
    }

    else if (ui.radioButtonRef3->isChecked())
    {
        if (m_iFileCnt < 4)
        {
            m_iRefImageIdx = 0;
            ui.radioButtonRef0->setChecked(true);
        }
        else //normal
            m_iRefImageIdx = 3;
    }
    else if (ui.radioButtonRef4->isChecked())
    {
        if (m_iFileCnt < 5)
        {
            m_iRefImageIdx = 0;
            ui.radioButtonRef0->setChecked(true);	
        }
        else //normal
            m_iRefImageIdx = 4;
    }
	
    ui.lineEditNorm0->clear();
    ui.lineEditNorm1->clear();
    ui.lineEditNorm2->clear();
    ui.lineEditNorm3->clear();
    ui.lineEditNorm4->clear();
}

void CrossRemoval::SLT_NormCalc()
{
    //Norm of Ref image should be always 1.0
    if (m_iFileCnt < 2)
        return;

    if (m_iRefImageIdx < 0)
        return;

    int size = m_iWidth * m_iHeight;

    ui.lineEditNorm0->setText(QString("%1").arg(1.000));
    ui.lineEditNorm1->setText(QString("%1").arg(1.000));
    ui.lineEditNorm2->setText(QString("%1").arg(1.000));
    ui.lineEditNorm3->setText(QString("%1").arg(1.000));
    ui.lineEditNorm4->setText(QString("%1").arg(1.000));

    //Calculate pixel ration: condition-> both ref and additional Image pixle should not be cross-hair
    for (int i = 0 ; i< m_iFileCnt ; i++)
    {
        if (i == m_iRefImageIdx)		
            continue;		

        double ratioSum = 0.0;
        int calcCnt = 0;

        double curRatio = 0.0;

        for (int k = 0 ;k < size; k++)
        {	
            if (m_arrYKImageMask[i].m_pData[k]== MASK_NONCROSSHAIR && m_arrYKImageMask[m_iRefImageIdx].m_pData[k]== MASK_NONCROSSHAIR)
            {
                if (m_arrYKImage[m_iRefImageIdx].m_pData[k] != 0) //Or !Bad pixel
                {
                    curRatio =  (m_arrYKImage[i].m_pData[k]) / (double)(m_arrYKImage[m_iRefImageIdx].m_pData[k]);
                    ratioSum = ratioSum + curRatio;
                    calcCnt++;
                }				
            }			
        }		
        double meanRatio = ratioSum / (double)calcCnt;

        if (i == 0)
        {
            ui.lineEditNorm0->setText(QString("%1").arg(meanRatio));
        }		
        else if (i == 1)
        {
            ui.lineEditNorm1->setText(QString("%1").arg(meanRatio));
        }
        else if (i == 2)
        {
            ui.lineEditNorm2->setText(QString("%1").arg(meanRatio));
        }
        else if (i == 3)
        {
            ui.lineEditNorm3->setText(QString("%1").arg(meanRatio));
        }
        else if (i == 4)
        {
            ui.lineEditNorm4->setText(QString("%1").arg(meanRatio));
        }
    }
}

void CrossRemoval::SLT_Normalization()
{
    for (int i = 0 ; i< m_iFileCnt ; i++)
    {
        if (i == m_iRefImageIdx)		
            continue;

        double corrF = 1.0;
        double denometer = 0.0;
        if (i == 0)
        {
            denometer =  ui.lineEditNorm0->text().toDouble();
            if (denometer > 0)
            {
                corrF = 1.0 / ui.lineEditNorm0->text().toDouble();
                m_arrYKImage[i].PixelMultiply(corrF);
            }
            else
                printf("Error! denometer is less than 0 image idx = %d\n", i);
        }		
        else if (i == 1)
        {
            denometer =  ui.lineEditNorm1->text().toDouble();
            if (denometer > 0)
            {
                corrF = 1.0 / ui.lineEditNorm1->text().toDouble();
                m_arrYKImage[i].PixelMultiply(corrF);
            }
            else
                printf("Error! denometer is less than 0 image idx = %d\n", i);			
        }
        else if (i == 2)
        {
            denometer =  ui.lineEditNorm2->text().toDouble();
            if (denometer > 0)
            {
                corrF = 1.0 / ui.lineEditNorm2->text().toDouble();
                m_arrYKImage[i].PixelMultiply(corrF);
            }
            else
                printf("Error! denometer is less than 0 image idx = %d\n", i);
        }
        else if (i == 3)
        {
            denometer =  ui.lineEditNorm3->text().toDouble();
            if (denometer > 0)
            {
                corrF = 1.0 / ui.lineEditNorm3->text().toDouble();
                m_arrYKImage[i].PixelMultiply(corrF);
            }
            else
                printf("Error! denometer is less than 0 image idx = %d\n", i);
        }
        else if (i == 4)
        {
            denometer =  ui.lineEditNorm4->text().toDouble();
            if (denometer > 0)
            {
                corrF = 1.0 / ui.lineEditNorm4->text().toDouble();
                m_arrYKImage[i].PixelMultiply(corrF);
            }
            else
                printf("Error! denometer is less than 0 image idx = %d\n", i);
        }
    }

    SLT_NormCalc();
}

void CrossRemoval::GeneratePixMedianImg( YK16GrayImage* pTargetImage, int arrSize, YK16GrayImage* arrImg, YK16GrayImage* arrImgMask, YK16GrayImage* arrImgReplaced, int refIdx )
{
    if (arrSize < 1)
        return;

    if (pTargetImage == NULL || pTargetImage->IsEmpty())
        return;

    int size = pTargetImage->m_iWidth * pTargetImage->m_iHeight;

    int sampleCnt = 0;

    int validSampleIdx0 = 0; //should be 0
    int validSampleIdx1 = 0;
    int validSampleIdx2 = 0;
    int validSampleIdx3 = 0;
    int validSampleIdx4 = 0;
    int validSampleIdx5 = 0;

    for (int i = 0 ; i<size ; i++)
    {
        pTargetImage->m_pData[i] = GetMedianPixValueFromMultipleImg(i, arrSize, arrImg, arrImgMask, arrImgReplaced, refIdx, &sampleCnt );

        if (sampleCnt == 0)
            validSampleIdx0++;
        else if (sampleCnt == 1)
            validSampleIdx1++;
        else if (sampleCnt == 2)
            validSampleIdx2++;
        else if (sampleCnt == 3)
            validSampleIdx3++;
        else if (sampleCnt == 4)
            validSampleIdx4++;
        else if (sampleCnt == 5)
            validSampleIdx5++;		
    }

    printf("Number of Samples for Median Selection\n Sample_0 = %d \n Sample_1 = %d \n Sample_2 = %d \n Sample_3 = %d \n Sample_4 = %d \n Sample_5 = %d \n",
        validSampleIdx0,
        validSampleIdx1,
        validSampleIdx2,
        validSampleIdx3,
        validSampleIdx4,
        validSampleIdx5
    );
}

unsigned short CrossRemoval::GetMedianPixValueFromMultipleImg( int pxIdx, int arrSize, YK16GrayImage* arrImg, YK16GrayImage* arrImgMask, YK16GrayImage* arrReplacedImg,int iRefIdx, int* sampleCnt )
{
    if (arrSize < 1)
    {
        (*sampleCnt) = 0;
        return 0;
    }
    if (iRefIdx >= arrSize || iRefIdx < 0 )
        return 0;

    vector<unsigned short> vValidPixVal;

    for (int k = 0 ; k < arrSize ; k++)
    {
        if (arrImgMask[k].m_pData[pxIdx] !=MASK_CROSSHAIR)
        {
            vValidPixVal.push_back(arrImg[k].m_pData[pxIdx]);
        }
    }
    int tmpSampleCnt = vValidPixVal.size();
    (*sampleCnt) = tmpSampleCnt	;

    sort (vValidPixVal.begin(), vValidPixVal.end()); //default -> ascending order

    /*if (pxIdx == 30000)
      {
      for (int i = 0 ; i<tmpSampleCnt ; i++)
      {
      printf("%d\n",vValidPixVal.at(i));
      }
      }*/

    if (tmpSampleCnt < 1)//if there is no valid point (all are in cross-hair region) --> get replaced image pixel
    {
	//	printf("Replaced pixel. Undesirable!\n");
        return arrReplacedImg[iRefIdx].m_pData[pxIdx];		
    }
    else if (tmpSampleCnt ==1)//if only single image has valid point on that position.
        return vValidPixVal.at(0);
    else if (tmpSampleCnt == 2)//if thre are 2 pixels
    {
        unsigned short mean =  (unsigned short)((vValidPixVal.at(0) +  vValidPixVal.at(1)) / 2.0);
        return mean;
    }
    else if (tmpSampleCnt > 2)//if thre are 2 pixels
    {
        int medianIdx = (int)(tmpSampleCnt / 2.0); //ROUND_LOWER
	/*	if (pxIdx == 30000)
		{		
                printf("SampleCnt: %d, medianIdx: %d \n",tmpSampleCnt, medianIdx);		
		}*/
        return vValidPixVal.at(medianIdx);
    }
    return 0;
}

void CrossRemoval::SLT_PixMedianImg()
{
    //GeneratePixMedianImg( m_pPixelMedianImage, m_iFileCnt, m_arrYKImage, m_arrYKImageMask, m_arrYKImageReplaced, m_iRefImageIdx);
    //m_pPixelMedianImage->SaveDataAsRaw("C:\\PixMedianImg.raw");

    //Gen2x2BinImg (&m_arrYKImage[0], &m_arrYKImage[1],m_pPixelMedianImage);
    //m_pPixelMedianImage->SaveDataAsRaw("C:\\Pix4SectorImage.raw");
}

void CrossRemoval::Gen2x2BinImg( YK16GrayImage* pImage14, YK16GrayImage* pImage23, YK16GrayImage* pTarImg )
{
    int width = pImage14->m_iWidth;
    int height = pImage14->m_iHeight;

    int i,j;
    //Sector1
    for (i = 0 ; i< (int)(height/2.0) ; i++)
    {
        for (j = 0 ; j < (int)(width/2.0); j++)
        {
            pTarImg->m_pData[width*i +j] = pImage14->m_pData[width*i +j];
        }
    }
    //Sector4
    for (i = (int)(height/2.0) ; i< height ; i++)
    {
        for (j = (int)(width/2.0) ; j < width; j++)
        {
            pTarImg->m_pData[width*i +j] = pImage14->m_pData[width*i +j];
        }
    }

    //Sector2
    for (i = 0 ; i< (int)(height/2.0) ; i++)
    {
        for (j = (int)(width/2.0) ; j < width; j++)
        {
            pTarImg->m_pData[width*i +j] = pImage23->m_pData[width*i +j];
        }
    }
    //Sector3
    for (i = (int)(height/2.0) ; i< height ; i++)
    {
        for (j = 0 ; j < (int)(width/2.0); j++)
        {
            pTarImg->m_pData[width*i +j] = pImage23->m_pData[width*i +j];
        }
    }
}
//
//void CrossRemoval::VertLineBasedReplacement( YK16GrayImage* pSrcReplaced,YK16GrayImage* pSrcMask, YK16GrayImage* pRef,YK16GrayImage* pRefMask, YK16GrayImage* pTarImg )
//{
//	int width = pSrcReplaced->m_iWidth;
//	int height = pSrcReplaced->m_iHeight;
//	
//	int i,j;
//
//	unsigned short* VertLineSrc = new unsigned short [height];
//	unsigned short* VertLineSrcMask = new unsigned short [height];
//	unsigned short* VertLineRef = new unsigned short [height];
//	unsigned short* VertLineRefMask = new unsigned short [height];
//	//unsigned short* VertLineConverted = new unsigned short [height];
//
//	for (j = 0 ; j<width ; j++)
//	{
//		for (i = 0 ; i< height ; i++)
//		{
//			VertLineSrc[i] = pSrcReplaced->m_pData[width*i+j];
//			VertLineSrcMask[i] = pSrcMask->m_pData[width*i+j];
//			VertLineRef[i] = pRef->m_pData[width*i+j];
//			VertLineRefMask[i] = pRefMask->m_pData[width*i+j];
//		}
//
//		PerLineReplacement(height, VertLineSrc, VertLineSrcMask, VertLineRef, VertLineRefMask); //only VertLineSrc will be changed
//
//		for (i = 0 ; i< height ; i++)
//		{
//			pSrcReplaced->m_pData[width*i+j] = VertLineSrc[i];
//		}		
//	}
//
//	delete [] VertLineSrc;
//	delete [] VertLineSrcMask;
//	delete [] VertLineRef;
//	delete [] VertLineRefMask;
//}
////1D Array replacement
//void CrossRemoval::PerLineReplacement( int rowSize, unsigned short* VertLineSrc, unsigned short* VertLineSrcMask, unsigned short* VertLineRef, unsigned short* VertLineRefMask )
//{
//	int i = 0;
//	
//	int prevVal  = -1;
//	int curVal  = -1;
//
//	for (i = 0 ; i< rowSize ; i++)
//	{
//		curVal = VertLineSrcMask[i];
//		//if ()
//		prevVal = curVal;
//	}
//}

void CrossRemoval::SLT_SaveMultipleMask()
{
    if (m_iFileCnt < 1)
        return;

    for (int i = 0 ; i<m_iFileCnt ; i++)
    {
        SaveAutoFileName(&m_arrYKImageMask[i], (QString&)(m_strListSrcFilePath.at(i)), "_MSK");
    }

    for (int i = 0 ; i<m_iFileCnt ; i++)
    {
        SaveAutoFileName(&m_arrYKImageReplaced[i], (QString&)(m_strListSrcFilePath.at(i)), "_CR");
    }

}

void CrossRemoval::SLT_SaveMultipleReplaced()
{

}

//
//void CrossRemoval::SLT_LoadRefImage()
//{
////		1) Open Raw File
//	QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Raw image file (*.raw)", 0,0);		
//
//	if (!m_pRefImage->LoadRawImage(fileName.toLocal8Bit().constData(),m_iWidth,m_iHeight))
//		return;
//
//	//m_strSrcFilePath = fileName;
//	//m_pImageYKCur->CopyFromBuffer(m_pImageYKSrc->m_pData, m_pImageYKSrc->m_iWidth, m_pImageYKSrc->m_iHeight);
//
//	/*double mean;
//	double SD;
//	double max;
//	double min;
//	m_pImageYKCur->CalcImageInfo(mean, SD, max, min);*/
//
//	/*ui.sliderMin->setValue((int)(mean - 4*SD));
//	ui.sliderMax->setValue((int)(mean + 4*SD));*/
//
//	SLT_DrawCurImage();	
//
//}
