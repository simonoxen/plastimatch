/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <QtGui>
#include <QFileDialog>
#include <QGraphicsScene>
#include <QPixmap>
#include <QGraphicsPixmapItem>
#include <QDir>
#include <QTimer>
#include <QRect>
#include "iqt_video_widget.h"
#include <QString>
#include <QTime>
#include <QRubberBand>
#include <QMouseEvent>
#include <QWheelEvent>
#include "sleeper.h"

Iqt_video_widget::Iqt_video_widget (QWidget *parent)
    : QGraphicsView (parent)
{
    scene = new QGraphicsScene;
    this->setScene (scene);
    pmi = new QGraphicsPixmapItem (
        QPixmap("/home/willemro/src/plastimatch/src/ise/test1.png"));
    scene->addItem(pmi);
    ping_pong = 0;
    qp1 = qp2 = 0;
    
    qp1 = new QPixmap;

    ping_check = new QTimer (this);
    connect (ping_check, SIGNAL(timeout()), this, SLOT(flick()));
    ping_check->start(500);
    
    SetCenter(QPointF(500.0, 500.0));
    //QGraphicsRectItem *rect_item 
      //  = scene->addRect (QRectF (20, 20, 10, 10));
    show();
}

void Iqt_video_widget::SetCenter (const QPointF& centerPoint) {
    QRectF visibleArea = mapToScene(rect()).boundingRect();
    QRectF sceneBounds = sceneRect();
    double boundX = visibleArea.width() / 2.0;
    double boundY = visibleArea.height() / 2.0;
    double boundwidth = sceneBounds.width() - 2.0 * boundX;
    double boundheight = sceneBounds.height() - 2.0 * boundY;
    QRectF bounds(boundX, boundY, boundwidth, boundheight);

    if(bounds.contains(centerPoint)) {
        this->currentCenter = centerPoint;
    } else {
        if(visibleArea.contains(sceneBounds)) {
	    this->currentCenter = sceneBounds.center();
	} else {
	    currentCenter = centerPoint;
	    if (centerPoint.x() > bounds.x() + bounds.width()) {
	      currentCenter.setX(bounds.x() + bounds.width());
	    } else if (centerPoint.x() < bounds.x()) {
	      currentCenter.setX(bounds.x());
	    }
	    if (centerPoint.y() > bounds.y() + bounds.height()) {
	      currentCenter.setY(bounds.y() + bounds.height());
	    } else if (centerPoint.y() < bounds.y()) {
	      currentCenter.setY(bounds.y());
	    }
	}
	centerOn(currentCenter);
    }
}

void Iqt_video_widget::mousePressEvent(QMouseEvent* event)
{
    this->origin = event->pos();
    if (!rubberband)
    {
        rubberband = new QRubberBand(QRubberBand::Rectangle, this);
	this->drawing = true;
    }
    rubberband->setGeometry(QRect(origin, QSize()));
    rubberband->show();
}

void Iqt_video_widget::mouseMoveEvent(QMouseEvent* event)
{
    if (drawing)
    {
        rubberband->setGeometry(QRect(origin, event->pos()).normalized());
    }
}

void Iqt_video_widget::mouseReleaseEvent(QMouseEvent* event)
{
    // if (rubberband->width() > rubberband->height())
    // {
    //     double scalefactor = this->width()/rubberband->width();
    //     scale(scalefactor, scalefactor);
    // } else {
    //     double scalefactor = this->height()/rubberband->height();
    // 	scale(scalefactor, scalefactor);
    // }
    qDebug("Top: %d\nRight: %d\nWidth: %d\nHeight: %d", origin.x(), origin.y(), rubberband->width(), rubberband->height());
    this->fitInView(origin.x(), origin.y(), rubberband->width(), rubberband->height(), Qt::KeepAspectRatio);
    rubberband->hide();
}

void Iqt_video_widget::load(const QString& filename) {
    
    qp1 = new QPixmap (filename);
    QString filename_2 = QFileInfo (filename).path() + "/test2.png";
    qp2 = new QPixmap (filename_2);//load new image

    this->filename = filename;

    // qDebug("Time Elapsed: %d ms", time->elapsed()); //display time in shell
    // QTime ntime(0,0,0,0);                   //initialize time to be displayed
    // j = time->elapsed();                    //msecs since time->start()
    //   ntime=ntime.addMSecs(j);                //time to be displayed
    //  QString text = ntime.toString("ss.zzz");//convert to text
    //  QMessageBox::information (0, QString ("Timer"), 
    //			QString ("Took %1 seconds").arg(text)); //display text
    // delete time;
    // delete qp1;
    // delete qp2;
} //end load()

void Iqt_video_widget::wheelEvent(QWheelEvent* event)
{
    QPointF pointBeforeScale(mapToScene(event->pos()));
    QPointF screenCenter = GetCenter();
    double scaleFactor = 1.15;
    if (event->delta() > 0)
    {
        scale(scaleFactor, scaleFactor);
    } else {
        scale(1.0/scaleFactor, 1.0/scaleFactor);
    }
    QPointF pointAfterScale(mapToScene(event->pos()));
    QPointF offset = pointBeforeScale - pointAfterScale;
    QPointF newCenter = screenCenter + offset;
    SetCenter (newCenter);
}

void Iqt_video_widget::flick(void)
{
    if (!filename.isNull()) {
        delete pmi;                       //remove old pmi (IS necessary)
        if (ping_pong == 0) {
            pmi = new QGraphicsPixmapItem(*qp1);//set loaded image as new pmi
            ping_pong = 1;
        } else {
            pmi = new QGraphicsPixmapItem(*qp2);//set loaded image as new pmi
            ping_pong = 0;
        }
        scene->addItem(pmi);                //add new image to scene

    } else {
        return;
    }
}

void Iqt_video_widget::stop ()
{
    ping_check->stop();
}

void Iqt_video_widget::play (bool playing)
{   
    if (playing) {
        ping_check->start(500);
    } else {
        ping_check->stop();
    }
}

void Iqt_video_widget::synth ()
{
    
}

Iqt_video_widget::~Iqt_video_widget ()
{
    delete qp1;
    delete qp2;
    delete pmi;
    delete ping_check;
}

void 
Iqt_video_widget::set_qimage (const QImage& qimage)
{
    delete pmi;                       //remove old pmi (IS necessary)
    
    pmi = new QGraphicsPixmapItem(QPixmap::fromImage (qimage.scaled(this->size(), Qt::KeepAspectRatio)));
    scene->addItem(pmi);
    scene->addText("Synthetic Fluoroscopy");
}
