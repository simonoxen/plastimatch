#ifndef QYKLABEL_H
#define QYKLABEL_H

#include <QLabel>
#include <QMouseEvent>
#include <QRect>
#include <vector>
//#include <QDebug>
class YK16GrayImage;

using namespace std;
class qyklabel : public QLabel
{
	Q_OBJECT

public:
	YK16GrayImage* m_pYK16Image;
	QRect m_Rt;
	std::vector<QPoint> m_vPt;
	bool m_bDrawPoints;

public:
	qyklabel(QWidget *parent);
	~qyklabel();

	void mouseMoveEvent(QMouseEvent *ev); // virtual function reimplementation
	void mousePressEvent(QMouseEvent *ev);
	void leaveEvent(QEvent *);
	int x,y;

	void SetBaseImage(YK16GrayImage* pYKImage);
	//void ConvertAndCopyPoints(vector<QPoint>& vSrcPoint);
	void ConvertAndCopyPoints(vector<QPoint>& vSrcPoint, int iDataWidth, int iDataHeight);

	QPoint View2Data(QPoint viewPt, int viewWidth, int viewHeight, int dataWidth, int dataHeight);
	QPoint Data2View(QPoint dataPt, int viewWidth, int viewHeight, int dataWidth, int dataHeight);


protected:
	void paintEvent(QPaintEvent *);

signals:
	void Mouse_Pressed();
	void Mouse_Pos();
	void Mouse_Left();

public slots:
	void SetDrawPointToggle(bool bToggle);

private:
	
};

#endif // QYKLABEL_H
