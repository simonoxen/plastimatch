#pragma once

class QPixmap;
class QLabel;

#define DEFAULT_WINLEVEL_MID 2047
#define DEFAULT_WINLEVEL_WIDTH 4094

class YK16GrayImage
{	
public:
	YK16GrayImage(void);
	YK16GrayImage(int width, int height);
	~YK16GrayImage(void);

	int m_iWidth;
	int m_iHeight;

	unsigned short* m_pData; // 0 - 65535

	QPixmap* m_pPixmap;

	bool LoadRawImage(const char *filePath, int width, int height);
	bool CopyFromBuffer(unsigned short* pImageBuf, int width, int height);

	bool CreateImage(int width, int height, unsigned short usVal);

	bool FillPixMap(int winMid, int winWidth);

	bool SaveDataAsRaw (const char *filePath);
	bool DrawToLabel(QLabel* lbDisplay);

	bool IsEmpty();
	bool ReleaseBuffer();

	
	//should be implemented later

	//Flip
	//mirror
	//rotation
	
};
