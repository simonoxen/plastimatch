#include "nki2mha_converter.h"
#include <QString>
#include <QFileDialog>
#include <QListView>
//#include "YK16GrayImage.h"
#include <fstream>

#include "mha_io.h"
#include "nki_io.h"
#include "volume.h"

nki2mha_converter::nki2mha_converter(QWidget *parent, Qt::WFlags flags)
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

nki2mha_converter::~nki2mha_converter()
{
	//delete m_pImgOffset;
	//delete m_pImgGain;

	//m_vPixelReplMap.clear(); //not necessary

}

void nki2mha_converter::SLT_OpenMultipleRaw()
{
	QStringList tmpList = QFileDialog::getOpenFileNames(this,"Select one or more files to open","/home","NKI Images (*.scan)");

	int iFileCnt = tmpList.size();

	if (iFileCnt < 1)
		return;

	m_strlistPath = tmpList;

	ui.plainTextEdit_Raw->clear();
	ui.plainTextEdit_Corrected->clear();

	for (int i = 0 ; i<iFileCnt ; i++)
	{
		ui.plainTextEdit_Raw->appendPlainText(m_strlistPath.at(i)); //just for display
	}

	//ui.listView_Raw->
	//ui.plainTextEdit_Raw
	//QPlainTextEdit* text;
}
//
QString nki2mha_converter::CorrectSingleFile(const char* filePath)
{
	Volume *v = nki_load (filePath);
	if (!v)
	{
		printf("file reading error\n");	
		return "";
	}	

	//Load raw file		
	//filePath
	//QString exportName = filePath;
	//corrImg.SaveDataAsRaw();
	QString endFix = "_CONV";

	QFileInfo srcFileInfo = QFileInfo(filePath);
	QDir dir = srcFileInfo.absoluteDir();
	QString baseName = srcFileInfo.completeBaseName();
	QString extName = "mha";

	QString newFileName = baseName.append(endFix).append(".").append(extName);
	QString newPath = dir.absolutePath() + "\\" + newFileName;	

	write_mha(newPath.toLocal8Bit().constData(), v);	

	return newPath;
	//corrImg.ReleaseBuffer();
}
//
void nki2mha_converter::SLT_Correct()
{
	//1) Load files from m_strlistPath
	//2) offset or gain correction
	//3) bad pixel correction if available
	int listSize = m_strlistPath.size();

	if (listSize < 1)
		return;

	for (int i = 0 ; i<listSize ; i++)
	{
		QString filePath = m_strlistPath.at(i);
		QString corrFilePath = CorrectSingleFile(filePath.toLocal8Bit().constData());
		ui.plainTextEdit_Corrected->appendPlainText(corrFilePath);
	}

}
//
//
//
//void nki2mha_converter::LoadBadPixelMap(const char* filePath)
//{
//	m_vPixelReplMap.clear();
//
//	ifstream fin;
//	fin.open(filePath);
//
//	if (fin.fail())
//		return;
//
//	char str[MAX_LINE_LENGTH];
//	//memset(str, 0, MAX_LINE_LENGTH);
//
//	while (!fin.eof())
//	{
//		memset(str, 0, MAX_LINE_LENGTH);
//		fin.getline(str, MAX_LINE_LENGTH);
//		QString tmpStr = QString(str);
//
//		if (tmpStr.contains("#ORIGINAL_X"))
//			break;
//	}
//
//	while (!fin.eof())
//	{
//		memset(str, 0, MAX_LINE_LENGTH);
//		fin.getline(str, MAX_LINE_LENGTH);
//		QString tmpStr = QString(str);
//
//		QStringList strList = tmpStr.split("	");
//
//		if (strList.size() == 4)
//		{
//			BADPIXELMAP tmpData;
//			tmpData.BadPixX = strList.at(0).toInt();
//			tmpData.BadPixY = strList.at(1).toInt();
//			tmpData.ReplPixX = strList.at(2).toInt();
//			tmpData.ReplPixY = strList.at(3).toInt();
//			m_vPixelReplMap.push_back(tmpData);
//		}	
//	}
//	fin.close();
//}
//
//void nki2mha_converter::BadPixReplacement(YK16GrayImage* targetImg)
//{
//	if (m_vPixelReplMap.empty())
//		return;	
//
//	int oriIdx, replIdx;
//
//	vector<BADPIXELMAP>::iterator it;
//
//	for (it = m_vPixelReplMap.begin() ; it != m_vPixelReplMap.end(); it++)
//	{
//		BADPIXELMAP tmpData= (*it);
//		oriIdx = tmpData.BadPixY * IMG_WIDTH + tmpData.BadPixX;
//		replIdx = tmpData.ReplPixY * IMG_WIDTH + tmpData.ReplPixX;
//		targetImg->m_pData[oriIdx] = targetImg->m_pData[replIdx];
//	}	
//}