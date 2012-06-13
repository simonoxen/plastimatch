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

    //QGraphicsRectItem *rect_item 
      //  = scene->addRect (QRectF (20, 20, 10, 10));
    show();
}

void Iqt_video_widget::load(const QString& filename) {
    time = new QTime();						//express new time variable
    QPixmap *qp1 = new QPixmap (filename);	//load new image
    QString filename_2 = QFileInfo (filename).path() + "/test2.png";
    QPixmap *qp2 = new QPixmap (filename_2);	//load new image
    time->start();							//start time

    int ping_pong = 0;
    for (int i = 0; i < 100; i++) {
    	delete pmi;							//remove old pmi (necessary?)
        if (ping_pong == 0) {
            pmi = new QGraphicsPixmapItem(*qp1);	//set loaded image as new pmi
            ping_pong = 1;
        } else {
            pmi = new QGraphicsPixmapItem(*qp2);	//set loaded image as new pmi
            ping_pong = 0;
        }
        scene->addItem(pmi);				//add new image to scene

        Sleeper::msleep (33);

    }
    qDebug("Time Elapsed: %d ms", time->elapsed()); //display time in shell
    QTime ntime(0,0,0,0);  					//initialize time to be displayed
    i = time->elapsed();					//msecs since time->start()
    ntime=ntime.addMSecs(i);				//time to be displayed
    QString text = ntime.toString("ss.zzz");//convert to text
    QMessageBox::information (0, QString ("Timer"), 
    				QString ("Took %1 seconds").arg(text)); //display text
    delete time;
    delete qp1;
    delete qp2;
} //end load()

Iqt_video_widget::~Iqt_video_widget ()
{
    delete pmi;
}
