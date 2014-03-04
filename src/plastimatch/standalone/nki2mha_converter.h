#ifndef NKI2MHA_CONVERTER_H
#define NKI2MHA_CONVERTER_H
//#ifndef nki2mha_converter_H
//#define nki2mha_converter_H

#include <QtGui/QMainWindow>
#include "ui_nki2mha_converter.h"
#include <QStringList>
#include <vector>
//#include "acquire_4030e_define.h"

//class YK16GrayImage;

//#define IMG_WIDTH 2304
//#define IMG_HEIGHT 3200


using namespace std;

//struct BADPIXELMAP{
//	int BadPixX;
//	int BadPixY;
//	int ReplPixX;
//	int ReplPixY;
//};
//
//struct PIXINFO{
//	int infoX;
//	int infoY;
//	unsigned short pixValue;
//};



class nki2mha_converter : public QMainWindow
{
	Q_OBJECT

public:
	nki2mha_converter(QWidget *parent = 0, Qt::WFlags flags = 0);
	~nki2mha_converter();
	QString CorrectSingle_NKI2MHA(const char* filePath);
	QString CorrectSingle_NKI2DCM(const char* filePath);
	QString CorrectSingle_NKI2RAW( const char* filePath );

	QString CorrectSingle_MHA2DCM(const char* filePath );
	//void LoadBadPixelMap(const char* filePath);
	//void BadPixReplacement(YK16GrayImage* targetImg);
	//void SaveBadPixelMap(vector<BADPIXELMAP>& vBadPixels);

	public slots:
		//void SLT_OpenOffsetFile();
		//void SLT_OpenGainFile();
		//void SLT_OpenBadpixelFile();
		void SLT_OpenMultipleRaw();
		void SLT_Correct_NKI2MHA(); //NKI to MHA
		void SLT_Correct_NKI2DCM(); //NKI to MHA
		void SLT_Correct_NKI2RAW(); //NKI to RAW: signed short!

public:
	//YK16GrayImage* m_pImgOffset;
	//YK16GrayImage* m_pImgGain;
	//Badpixmap;
	
	//vector<BADPIXELMAP> m_vPixelReplMap;	
	//vector<YK16GrayImage*> m_vpRawImg;
	QStringList m_strlistPath;

	


private:
	Ui::nki2mha_converterClass ui;
};

#endif // nki2mha_converter_H
