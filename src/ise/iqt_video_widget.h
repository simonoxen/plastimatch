/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _iqt_video_widget_h_
#define _iqt_video_widget_h_

#include <QGraphicsView>
#include <QPixmap>
#include <QtGui/QGraphicsView>
#include <QGraphicsScene>

class Iqt_video_widget : public QGraphicsView {
    Q_OBJECT

private:

	QGraphicsScene* scene;
	QPixmap pmap;
	QGraphicsPixmapItem* pmi;

public:
    Iqt_video_widget (QWidget *parent = 0);
    ~Iqt_video_widget ();

protected:
//    void paintEvent (QPaintEvent *event);

public:
    QPixmap m_pixmap;
};
#endif
