#include "YK16GrayImage.h"
#include <QPixmap>
#include <fstream>
#include <QLabel>
#include <QLabel>
#include "aqprintf.h"

YK16GrayImage::YK16GrayImage(void)
{
	m_iWidth = 0;
	m_iHeight = 0;

	m_pData = NULL;
	m_pPixmap = NULL;// for display

}

YK16GrayImage::YK16GrayImage(int width, int height)
{
	CreateImage(width, height, 0);		
}

YK16GrayImage::~YK16GrayImage(void)
{
	ReleaseBuffer();
}


bool YK16GrayImage::IsEmpty()
{
	if (m_pData == NULL)
		return true;
	else
		return false;
}

bool YK16GrayImage::CreateImage(int width, int height, unsigned short usVal)
{
	if (width < 1 || height < 1)
		return false;

	if (usVal < 0 || usVal > 65535)
		usVal = 0;

	//if (m_pData != NULL)
		//delete [] m_pData;

	ReleaseBuffer();

	m_iWidth = width;
	m_iHeight = height;

	int imgSize = width*height;
	m_pData = new unsigned short [imgSize];

	for (int i = 0 ; i<imgSize ; i++)
	{
		m_pData[i] = usVal;
	}	

	return true;
}


bool YK16GrayImage::LoadRawImage(const char *filePath, int width, int height)
{
	if (width < 1 || height < 1)
		return false;

	//if (m_pData != NULL)
	//	delete [] m_pData;	
	ReleaseBuffer();

	m_iWidth = width;
	m_iHeight = height;

	int imgSize = width*height;
	m_pData = new unsigned short [imgSize];	

	//aqprintf("ImageInfo in LoadRawImage, w: %d  h: %d   %d  %d \n",width, height, m_iWidth, m_iHeight);

	FILE* fd = NULL;
	fd = fopen(filePath, "rb");
	unsigned short buf = 0;	

	for (int i = 0 ; i<imgSize ; i++)
	{
		fread(&buf, 2, 1, fd);
		m_pData[i] = buf;
	}	

	fclose(fd);
	return true;
}

bool YK16GrayImage::CopyFromBuffer(unsigned short* pImageBuf, int width, int height)
{
	if (m_pData == NULL)
		return false;
	if (pImageBuf == NULL)
		return false;
	if (width != m_iWidth || height != m_iHeight)
		return false;

	int imgSize = m_iWidth*m_iHeight;

	for (int i = 0 ; i<imgSize ; i++)
	{
		m_pData[i] = pImageBuf[i];
	}
	return true;
}
bool YK16GrayImage::FillPixMap(int winMid, int winWidth) //0-65535 Сп window level
{	
	if (m_pData == NULL)
		return false;

	if (m_pPixmap != NULL)
	{
		//aqprintf("QPixmap already exist\n");
		delete m_pPixmap;
		m_pPixmap = NULL;
	}	
	m_pPixmap = new QPixmap(QSize(m_iWidth,m_iHeight)); //something happened here!!!: w: 4289140  h: 0	
	
	//8 bit gray buffer preparing

	int size = m_iWidth*m_iHeight;


	uchar* tmpData = new uchar [size*3];//RGB
	unsigned short uppVal = (int)(winMid + winWidth/2.0);
	unsigned short lowVal = (int)(winMid - winWidth/2.0);	

	//It takes 0.4 s in Release mode

	for (int i = 0 ; i<m_iHeight ; i++) //So long time....
	{
		for (int j = 0 ; j<m_iWidth ; j++)
		{
			int tmpIdx = 3*(i*m_iWidth+j);

			if (m_pData[i*m_iWidth+j] >= uppVal)
			{
				tmpData[tmpIdx+0] = 255;
				tmpData[tmpIdx+1] = 255;
				tmpData[tmpIdx+2] = 255;
			}
			else if (m_pData[i*m_iWidth+j] <= lowVal)
			{
				tmpData[tmpIdx+0] = 0;
				tmpData[tmpIdx+1] = 0;
				tmpData[tmpIdx+2] = 0;
			}
			else
			{
				tmpData[tmpIdx+0] = (uchar) ((m_pData[i*m_iWidth+j] - lowVal)/(double)winWidth * 255.0); //success
				tmpData[tmpIdx+1] = (uchar) ((m_pData[i*m_iWidth+j] - lowVal)/(double)winWidth * 255.0); //success
				tmpData[tmpIdx+2] = (uchar) ((m_pData[i*m_iWidth+j] - lowVal)/(double)winWidth * 255.0); //success
				//if (i<10000)
				//	fout << (int)(tmpData[i]) << std::endl;
			}
		}		
	}
	QImage tmpQImage = QImage((unsigned char*)tmpData,m_iWidth, m_iHeight,QImage::Format_RGB888);
	*m_pPixmap = QPixmap::fromImage(tmpQImage);

	delete [] tmpData;


	return true;
}

bool YK16GrayImage::SaveDataAsRaw (const char *filePath) //save 16 bit gray raw file
{
	if (m_pData == NULL)
		return false;

	int imgSize = m_iWidth*m_iHeight;		

	FILE* fd = NULL;
	fd = fopen(filePath, "wb");
	

	for (int i = 0 ; i<imgSize ; i++)
	{		
		fwrite(&m_pData[i], 2, 1, fd);		
	}	

	fclose(fd);
	return true;
}

bool YK16GrayImage::DrawToLabel( QLabel* lbDisplay )
{
	if (m_pPixmap == NULL)
		return false;

	int width = lbDisplay->width();
	int height = lbDisplay->height();
	//m_pPixmap->scaled(wid,showHeght,Qt::IgnoreAspectRatio)
	lbDisplay->setPixmap(m_pPixmap->scaled(width, height, Qt::IgnoreAspectRatio));
	return true;
}

bool YK16GrayImage::ReleaseBuffer()
{
	if (m_pData != NULL)
	{
		delete [] m_pData;
		m_pData = NULL;
	}
	if (m_pPixmap != NULL)
	{
		delete m_pPixmap;
		m_pPixmap = NULL;
	}
	m_iWidth = 0;
	m_iHeight = 0;

	return true;
}
