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

#include "iqt_video_widget.h"

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
//	scene = new QGraphicsScene;
//	this->setScene (scene);
//	QString file = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::currentPath());
    QPixmap *qp = new QPixmap (filename);
    for (int i = 0; i < 10; i++) {
        delete pmi;
        pmi = new QGraphicsPixmapItem(*qp);
	scene->addItem(pmi);
    }
    delete qp;
} //end load()

Iqt_video_widget::~Iqt_video_widget ()
{
    delete pmi;
}
