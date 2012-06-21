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
#include "iqt_video_widget.h"
#include <QString>
#include <QTime>
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
    
    //QGraphicsRectItem *rect_item 
      //  = scene->addRect (QRectF (20, 20, 10, 10));
    show();
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
    pmi = new QGraphicsPixmapItem(QPixmap::fromImage (qimage));
    scene->addItem(pmi);
    scene->addText("Synthetic Fluoroscopy");
}
