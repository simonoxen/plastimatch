/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <QtGui>
#include <QGraphicsScene>
#include <QPixmap>
#include <QGraphicsPixmapItem>

#include "iqt_video_widget.h"

Iqt_video_widget::Iqt_video_widget (QWidget *parent)
    : QGraphicsView (parent)
{
    scene = new QGraphicsScene;
    this->setScene (scene);

    pmi = new QGraphicsPixmapItem (
        QPixmap("/PHShome/gcs6/work/plastimatch/src/ise/test1.png"));
    scene->addItem(pmi);

    QGraphicsRectItem *rect_item 
        = scene->addRect (QRectF (20, 20, 10, 10));

    show();
}

Iqt_video_widget::~Iqt_video_widget ()
{
}
