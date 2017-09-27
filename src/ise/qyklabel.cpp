#include "qyklabel.h"
#include <QPainter>
#include "YK16GrayImage.h"

using namespace std;

qyklabel::qyklabel(QWidget *parent)
	: QLabel(parent)
{
	m_pYK16Image = NULL;
	//this->width();
	//m_Rt = this->rect();

	//m_Rt.setRect()
	m_bDrawPoints = true;

}

qyklabel::~qyklabel()
{
}

void qyklabel::mouseMoveEvent( QMouseEvent *ev )
{
	this->x	= ev->x();
	this->y = ev->y();
	emit Mouse_Pos();
}

void qyklabel::mousePressEvent( QMouseEvent *ev )
{
	this->x	= ev->x();
	this->y = ev->y();
	emit Mouse_Pressed();
}

void qyklabel::leaveEvent( QEvent * )
{
	emit Mouse_Left();
}

void qyklabel::paintEvent( QPaintEvent * )
{
	QPainter painter(this);
	
	painter.setPen( QPen(Qt::black, 2));
	QRect TargetRt = rect();
	//painter.drawPoint(100,100);
	painter.drawRect(TargetRt);

	if (m_pYK16Image != NULL)
	{
		//QRect imgSrcRect;
		//imgSrcRect.setRect(0,0,m_pYK16Image->m_iWidth, m_pYK16Image->m_iHeight);
		//painter.drawImage(rect(), m_pYK16Image->m_QImage,imgSrcRect,);

		//int width =  m_pYK16Image->m_QImage.width();
		//int height =  m_pYK16Image->m_QImage.height();

		//m_pYK16Image->m_QImage.save("C:\\111.png");

		//QImage tmpQImage = QImage("C:\\FromFillPixmap.png");

		//QImage tmpQImage = m_pYK16Image->m_QImage;
	
		//painter.drawImage(TargetRt, m_pYK16Image->m_QImage, imgSrcRect, QT::RGB888);
		//painter.drawImage(TargetRt, m_pYK16Image->m_QImage, imgSrcRect);
		painter.drawImage(TargetRt, m_pYK16Image->m_QImage); //it Works!
	}
	
	if (m_bDrawPoints)
	{
		painter.setPen( QPen(Qt::red, 2));
		vector<QPoint>::iterator it;
		for (it = m_vPt.begin() ; it != m_vPt.end() ; it++)
		{
			painter.drawPoint((*it).x(),(*it).y());
		}
	}	
}

void qyklabel::SetBaseImage( YK16GrayImage* pYKImage )
{
	if (pYKImage->m_pData != NULL && !pYKImage->m_QImage.isNull())
		m_pYK16Image = pYKImage;
}

void qyklabel::ConvertAndCopyPoints(vector<QPoint>& vSrcPoint, int iDataWidth, int iDataHeight)
{
	m_vPt.clear();	

	int dspWidth = this->width();
	int dspHeight = this->height();


	vector<QPoint>::iterator it;

	for (it = vSrcPoint.begin() ; it != vSrcPoint.end() ; it++)
	{
		if ((*it).x() == 381 && (*it).y() > 1000)
			int test = 5;

		if ((*it).x() == 581 && (*it).y() > 1000)
			int test = 5;


		QPoint tmpDspPt = Data2View((*it),dspWidth, dspHeight, iDataWidth, iDataHeight);
		m_vPt.push_back(tmpDspPt);
	}
}


QPoint qyklabel::View2Data(QPoint viewPt, int viewWidth, int viewHeight, int dataWidth, int dataHeight)
{
	double fZoomX = viewWidth / (double)dataWidth;
	double fZoomY = viewHeight / (double)dataHeight;

	QPoint dataPt;
	dataPt.setX(qRound(viewPt.x() / fZoomX));
	dataPt.setY(qRound(viewPt.y() / fZoomY));

	return dataPt;
}

QPoint qyklabel::Data2View(QPoint dataPt, int viewWidth, int viewHeight, int dataWidth, int dataHeight)
{
	double fZoomX = viewWidth / (double)dataWidth;
	double fZoomY = viewHeight / (double)dataHeight;

	QPoint viewPt;
	viewPt.setX(qRound(dataPt.x() * fZoomX));
	viewPt.setY(qRound(dataPt.y() * fZoomY));

	return viewPt;

}

void qyklabel::SetDrawPointToggle(bool bToggle )
{
	if (bToggle)
		m_bDrawPoints = true;
	else
		m_bDrawPoints = false;


	update();
}
