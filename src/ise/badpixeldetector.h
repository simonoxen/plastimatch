#ifndef BADPIXELDETECTOR_H
#define BADPIXELDETECTOR_H

#include <QtGui/QMainWindow>
#include "ui_badpixeldetector.h"
#include "YK16GrayImage.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDerivativeImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkMedianImageFilter.h"

#define DEFAULT_PERCENT_BADPIX_ON_COLUMN 30
#define DEFAULT_PERCENT_BADPIX_ON_ROW 30

typedef itk::Image<float,2> FloatImageType;
typedef itk::Image<unsigned short, 2> UnsignedShortImageType;
typedef itk::ImageFileReader<UnsignedShortImageType> readerType;
typedef itk::DerivativeImageFilter<UnsignedShortImageType, FloatImageType> DerivativeFilterType;
typedef itk::SmoothingRecursiveGaussianImageFilter<UnsignedShortImageType, UnsignedShortImageType>  SmoothingFilterType;
typedef itk::IntensityWindowingImageFilter<UnsignedShortImageType, UnsignedShortImageType>  WindowingFilterType;
typedef itk::MedianImageFilter<UnsignedShortImageType,UnsignedShortImageType> MedianFilterType;


using namespace std;


struct PIXINFO{
	int infoX;
	int infoY;
	unsigned short pixValue;
};



class BadPixelDetector : public QMainWindow
{
    Q_OBJECT
    ;
public:
    BadPixelDetector(QWidget *parent = 0, Qt::WFlags flags = 0);
    ~BadPixelDetector();
    void LoadBadPixelMap (const char* filePath);
    //direction == 0: hor bad pix row, directino ==1: ver bad pix column
    int DetectBadColumns (int direction);
    void AddBadColumn (int col);
    void SortAndRemoveDuplicates ();
    void FindReplacements ();
    void DetectBadPixels (bool bRefresh);

public slots:		
    void SLT_LoadDarkImage();
    void SLT_LoadGainImage();
    void SLT_DrawDarkImage();
    void SLT_DrawGainImage();
    void SLT_ShowBadPixels();
    void SLT_ResetMap();
    void SLT_DetectBadPixels();
    void SLT_SavePixelMap();
    void SLT_UncorrectDark(); //Gain Image + Dark
    void SLT_DoReplacement_Dark();
    void SLT_DoReplacement_Gain();		
    void SLT_LoadBadPixelMap();
    void SLT_SaveCurDark();
    void SLT_SaveCurGain();
    void SLT_AccumulateBadPixels();
    void SLT_AddManual();
    
    /*void SLT_GoToOriginal();}
      void SLT_Median();
      void SLT_Gaussian();
      void SLT_DerivativeHor();
      void SLT_DerivativeVer();		
      void SLT_RemoveCrosshairHor();
      void SLT_RemoveCrosshairVert();
      void SLT_SaveAs();
      void SLT_SetCurImageAsNewSourceImage();
      void SLT_RemoveCrosshairMacro();*/
    
    
public:
    //unsigned short GetSubstituteVert(int x,int y, double coeff0, double coeff1, int halfThickness, int iSamplePixels, unsigned short* medianImg, int width, int height);
    //unsigned short GetSubstituteHor(int x,int fixedY, double coeff0, double coeff1, int halfThickness, int iSamplePixels, unsigned short* medianImg, int width, int height);
    
    //bool SaveAutoFileName(QString& srcFilePath, QString endFix);
    
    
    
    
public:
    YK16GrayImage* m_pImageYKDark;
    YK16GrayImage* m_pImageYKGain;
    
    QString m_strSrcFilePathDark;
    QString m_strSrcFilePathGain;	
    
    //UnsignedShortImageType::Pointer m_itkCurImage;
    //FloatImageType::Pointer m_spTarImage;
    
    //UnsignedShortImageType::Pointer m_spTarImageDisp;
    int m_iWidth;
    int m_iHeight;
    
    vector<BADPIXELMAP> m_vPixelReplMap;	
    
    double m_fPercentThre;
    int m_iMedianSize;
    int m_sAddManual;

private:
    Ui::BadPixelDetectorClass ui;
};

#endif
