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
	scene = new QGraphicsScene (this);
	QGraphicsPixmapItem pmi(QPixmap("~/src/plastimatch/src/ise/test1.png"));
	scene->addItem(&pmi);
	show();
}

Iqt_video_widget::~Iqt_video_widget ()
{
}

void 
Iqt_video_widget::paintEvent (QPaintEvent *event)
{
}
