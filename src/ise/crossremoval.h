#ifndef CROSSREMOVAL_H
#define CROSSREMOVAL_H

#include <QtGui/QMainWindow>
#include "ui_crossremoval.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDerivativeImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkMedianImageFilter.h"
#include <QStringList>

class YK16GrayImage;

#define BLACK_VALUE 0
#define NORMAL_VALUE 1000
#define WHITE_VALUE 4095
#define CENTERLINE_VALUE 30000

#define DEFAULT_CROSSHAIR_MARGIN 20
#define DEFAULT_SAMPLING_PIXELS 10
#define DEFAULT_ROI_RATIO_X 0.5
#define DEFAULT_ROI_RATIO_Y 0.5

#define INCL_DIFF_THRESHOLD 0.3
#define DEFAULT_PIX_DIFF_PERCENT 30 //30%

#define MASK_CROSSHAIR 2
#define MASK_NONCROSSHAIR 0

struct IMAGEPROC_PARAM
{
	int MEDIAN_size;
	int GAUSSIAN_sigma;
	int	additionalMargin;
	double ROI_RatioX;
	double ROI_RatioY;
	double continuityThreshold;
};


struct CROSSHAIRINFO
{
	double GradientHor;
	double yCutHor;
	int thickenssHor;
	double GradientVer;
	double yCutVer;
	int thickenssVer;
	
	YK16GrayImage* pMaskImg;
};


typedef itk::Image<float,2> FloatImageType;
typedef itk::Image<unsigned short, 2> UnsignedShortImageType;
typedef itk::ImageFileReader<UnsignedShortImageType> readerType;
typedef itk::DerivativeImageFilter<UnsignedShortImageType, FloatImageType> DerivativeFilterType;
typedef itk::SmoothingRecursiveGaussianImageFilter<UnsignedShortImageType, UnsignedShortImageType>  SmoothingFilterType;
typedef itk::IntensityWindowingImageFilter<UnsignedShortImageType, UnsignedShortImageType>  WindowingFilterType;
typedef itk::MedianImageFilter<UnsignedShortImageType,UnsignedShortImageType> MedianFilterType;


class CrossRemoval : public QMainWindow
{
	Q_OBJECT

public:
	CrossRemoval(QWidget *parent = 0, Qt::WFlags flags = 0);
	~CrossRemoval();

	enum DIRECTION{
		HORIZONTAL = 0,
		VERTICAL
	};

	enum REPLACEOPTION{
		XY_FROM_MEDIAN = 0, //general use
		Y_FROM_ORIGINAL //less cross-hair artifact: Can be used for oblique cross-hair
	};
	bool MedianFiltering(YK16GrayImage* pImage, int medWindow);
	bool GaussianFiltering(YK16GrayImage* pImage, double sigma);
	//bool DerivativeFiltering(YK16GrayImage* pImage, int direction);
	//bool ReplacingCrosshairRegion (YK16GrayImage* pImage, double sigma, int direction);
	bool DerivativeFiltering( YK16GrayImage* pImage, int direction );
	//void GetLineEq(int direction, YK16GrayImage* pDerivativeImage, double* Grad, double* yCut, int* thickness, double ROI_RatioX, double ROI_RatioY); // 0:hor, 1: vert
	void GetLineEqFromDerivativeImg(int direction, YK16GrayImage* pDerivativeImage, double* Grad, double* yCut, int* thickness, double ROI_RatioX, double ROI_RatioY, double inclThreshold); // 0:hor, 1: vert
	void GenerateMaskImgForSingle(int direction, YK16GrayImage* pTargetMaskImg, int margin, double Grad, double yCut);//margin: half margin

	//void GetCrosshairMask(IMAGEPROC_PARAM& imgProcParam, CROSSHAIRINFO* crossInfo, YK16GrayImage* srcImg, YK16GrayImage* pTargetReplacedImg,  YK16GrayImage* pTargetMaskImg);
	void GetCrosshairMask(IMAGEPROC_PARAM& imgProcParam, CROSSHAIRINFO* crossInfo, YK16GrayImage* srcImg, YK16GrayImage* pTargetReplacedImg, YK16GrayImage* pTargetMaskImg, int ReplacementOption);
	
	//arrImg is supposed to be normalized first. but for the test purpose it can be also used w/o normalization
	//Median value should be selected among valid pixel only. In Mask, MASK_CROSSHAIR pixel will not be excluded in median calculation
	//void GeneratePixMedianImg (YK16GrayImage* pTargetImage, int arrSize, YK16GrayImage* arrImg, YK16GrayImage* arrImgMask);
	void GeneratePixMedianImg( YK16GrayImage* pTargetImage, int arrSize, YK16GrayImage* arrImg, YK16GrayImage* arrImgMask, YK16GrayImage* arrImgReplaced, int refIdx );
	//unsigned short GetMedianPixValueFromMultipleImg(int pxIdx, int arrSize, YK16GrayImage* arrImg, YK16GrayImage* arrImgMask, int* sampleCnt );
	unsigned short GetMedianPixValueFromMultipleImg( int pxIdx, int arrSize, YK16GrayImage* arrImg, YK16GrayImage* arrImgMask, YK16GrayImage* arrReplacedImg,int iRefIdx, int* sampleCnt );

	void Gen2x2BinImg( YK16GrayImage* pImage14, YK16GrayImage* pImage23, YK16GrayImage* pTarImg );

	//pSrc1: replacedImage (not original image)
	//void VertLineBasedReplacement(YK16GrayImage* pSrc1,YK16GrayImage* pMask1, YK16GrayImage* pSrc2,YK16GrayImage* pMask2, YK16GrayImage* pTarImg);
	//void PerLineReplacement(int rowSize, unsigned short* VertLineSrc, unsigned short* VertLineSrcMask, unsigned short* VertLineRef, unsigned short* VertLineRefMask); //only VertLineSrc will be changed

	public slots:
		//void SourceFileOpen();
		//void ReDrawSrc(); //내부적으로 slider의 값을 받아서 draw
		//void ReDrawTar();
		//void CopyYKImage2ItkImage(YK16GrayImage* pYKImage, UnsignedShortImageType::Pointer& m_spSrcImage);	
		//void CopyItkImage2YKImage(UnsignedShortImageType::Pointer& m_spSrcImage, YK16GrayImage* pYKImage);
		////Filter slots:
		////1) Copy m_pImageYKCur to m_spSrcImage(buffer is prepared already)
		////2) Filter: m_spSrcImage  --> tmpTarget Pointer (inside the filter)
		////3) Update m_pImageYKCur: Region iterator of tmpTarget Pointer -> m_pImageYKCur or (CopyItkImage2YKImage if type is same)
		////4) Redraw Target using ,_pCurYKImage
		//void FirstDerivative_Y();
		//void GaussianSmoothing();
		//void IntensityWindowing(); //get slider bar value from original image
		void SLT_LoadImage();//Load multipleimage
		//void SLT_LoadRefImage(); //Load single ref image

		void SLT_DrawCurImage();
		void SLT_GoToOriginal();
		void SLT_Median();
		void SLT_Gaussian();
		void SLT_DerivativeHor();
		void SLT_DerivativeVer();		
		void SLT_RemoveCrosshairHor();
		void SLT_RemoveCrosshairVert();
		void SLT_SaveAs();
		void SLT_SetCurImageAsNewSourceImage();

		void SLT_RemoveCrosshairMacro();
		void SLT_CrosshairDetection();
		void SLT_SetRefImage();

		void SLT_NormCalc();
		void SLT_Normalization();
		void SLT_PixMedianImg();
		void SLT_SaveMultipleMask();
		void SLT_SaveMultipleReplaced();

	
public:
	unsigned short GetSubstituteVert(int x,int y, double coeff0, double coeff1, int halfThickness, int iSamplePixels, unsigned short* medianImg, int width, int height);
	unsigned short GetSubstituteHor(int x,int fixedY, double coeff0, double coeff1, int halfThickness, int iSamplePixels, unsigned short* medianImg, int width, int height);

	bool SaveAutoFileName(QString& srcFilePath, QString endFix); //YKCur image default
	bool SaveAutoFileName(YK16GrayImage* pYK16Img, QString& srcFilePath, QString endFix);


	

public:
	YK16GrayImage* m_pImageYKSrc;
	YK16GrayImage* m_pImageYKCur; //	
	YK16GrayImage* m_pMedianYKImg;

	//YK16GrayImage* m_pMaskHor; //  crosshair = 1 , valid pixels = 0
	//YK16GrayImage* m_pMaskVert; // 0 and 1
	
	
	int m_iFileCnt;

	QString m_strSrcFilePath;
	//YK16GrayImage* m_pImageYKTar_MaxBin; //

	UnsignedShortImageType::Pointer m_itkCurImage;
	//FloatImageType::Pointer m_spTarImage;

	//UnsignedShortImageType::Pointer m_spTarImageDisp;
	int m_iWidth;
	int m_iHeight;

	int m_iCrosshairMargin;
	double m_fSearchROI_X; //if this value = 50, only cross-hair candidates in half region of the full image will be recognized as valid.
	double m_fSearchROI_Y;


	YK16GrayImage* m_pMaskComposite; //0 and 1
	YK16GrayImage* m_arrYKImage; //image array
	void ReleaseMemory();
	YK16GrayImage* m_arrYKImageMask; //image array	
	YK16GrayImage* m_arrYKImageReplaced; //image array	
	CROSSHAIRINFO* m_pCrosshairInfo;
	QStringList m_strListSrcFilePath;


	YK16GrayImage* m_pPixelMedianImage; // multiple image 에서 median 값으로 구성한 composite image. it will display statistic info. about median sample number.
	//this image should be prepared after normalization
	//this image can potentially be used for gain correction

		
	//YK16GrayImage* m_pRefImage;
	int m_iRefImageIdx;


private:
	Ui::CrossRemovalClass ui;
};

#endif // CROSSREMOVAL_H
