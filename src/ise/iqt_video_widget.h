/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _iqt_video_widget_h_
#define _iqt_video_widget_h_

#include <QGraphicsView>
#include <QPixmap>
#include <QtGui/QGraphicsView>
#include <QGraphicsScene>
#include <QTime>

class Iqt_video_widget : public QGraphicsView {
    Q_OBJECT

private:

    QGraphicsScene* scene;
    QPixmap pmap;
    QGraphicsPixmapItem* pmi;
    QTime *time;
    QTimer *ping_check;
    QString filename;
    QPixmap *qp1;
    QPixmap *qp2;
    int ping_pong;

public slots:
    void load(const QString& filename);
    void flick (void);
	//void addTime()

public:
    Iqt_video_widget (QWidget *parent = 0);
    int j;
    ~Iqt_video_widget ();

protected:
//    void paintEvent (QPaintEvent *event);

public:
    QPixmap m_pixmap;
};
#endif
