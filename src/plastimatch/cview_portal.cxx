#include <float.h>
#include <math.h>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QColor>
#include <QPainter>
#include <QWheelEvent>
#include <QGraphicsPixmapItem>
#include "volume.h"
#include "cview_portal.h"
#include "stdio.h"

// TODO: Maintain aspect ratio within portal

#define ROUND_INT(x) ((x)>=0?(long)((x)+0.5):(long)(-(-(x)+0.5)))

#if 0
static inline int
get_index (int* ij, int slice, int* stride)
{
    return stride[2] * slice
         + stride[1] * ij[1]
         + stride[0] * ij[0];
}
#endif

PortalWidget::PortalWidget (int width, int height)
{
    this->vol = NULL;
    this->slice = NULL;
    this->view = PV_AXIAL;
    this->dim[0] = width;
    this->dim[1] = height;

    this->scene = new QGraphicsScene (this);
    (this->scene)->setItemIndexMethod (QGraphicsScene::NoIndex);
    (this->scene)->setSceneRect (0, 0, width, height);
    (this->scene)->setBackgroundBrush (QBrush (Qt::black, Qt::SolidPattern));

    setScene (this->scene);
    setWindowTitle ("CrystalView v0.01a");
}

void
PortalWidget::setView (enum PortalViewType view)
{
    this->view = view;

    /* Delete the old rendering surface */
    if (this->slice) {
        (this->scene)->removeItem (pmi);
        delete this->slice;
    }

    /* Change coordinate systems */
    switch (this->view) {
    case PV_AXIAL:
        this->ijk_max[0] = (this->vol)->dim[0];
        this->ijk_max[1] = (this->vol)->dim[1];
        this->ijk_max[2] = (this->vol)->dim[2];

        this->stride[0] = 1;
        this->stride[1] = (this->vol)->dim[0];
        this->stride[2] = (this->vol)->dim[1] * (this->vol)->dim[0];

        this->spacing[0] = (this->vol)->spacing[0];
        this->spacing[1] = (this->vol)->spacing[1];

        this->offset[0] = (this->vol)->offset[0];
        this->offset[1] = (this->vol)->offset[1];
        break;
    case PV_CORONAL:
        this->ijk_max[0] = (this->vol)->dim[0];
        this->ijk_max[1] = (this->vol)->dim[2];
        this->ijk_max[2] = (this->vol)->dim[1];

        this->stride[0] = 1;
        this->stride[1] = (this->vol)->dim[1] * (this->vol)->dim[0];
        this->stride[2] = (this->vol)->dim[0];

        this->spacing[0] = (this->vol)->spacing[0];
        this->spacing[1] = (this->vol)->spacing[2];

        this->offset[0] = (this->vol)->offset[0];
        this->offset[1] = (this->vol)->offset[2];
        break;
    case PV_SAGITTAL:
        this->ijk_max[0] = (this->vol)->dim[1];
        this->ijk_max[1] = (this->vol)->dim[2];
        this->ijk_max[2] = (this->vol)->dim[0];

        this->stride[0] = (this->vol)->dim[0];
        this->stride[1] = (this->vol)->dim[1] * (this->vol)->dim[0];
        this->stride[2] = 1;

        this->spacing[0] = (this->vol)->spacing[1];
        this->spacing[1] = (this->vol)->spacing[2];

        this->offset[0] = (this->vol)->offset[1];
        this->offset[1] = (this->vol)->offset[2];
        break;
    default:
        exit (-1);
        break;
    }

    /* Portal resolution (mm per pix) */
    this->res[0] = (this->spacing[0] * (float)this->ijk_max[0]) / (float)this->dim[0];
    this->res[1] = (this->spacing[1] * (float)this->ijk_max[1]) / (float)this->dim[1];

    /* Make a new rendering surface */
    this->slice = new QImage (this->dim[0], this->dim[1], QImage::Format_RGB32);
    this->pmi = (this->scene)->addPixmap (this->pmap);
    this->renderSlice (this->ijk_max[2] / 2);
}

int
PortalWidget::getPixelValue (float hfu)
{
    int scaled = floor ((hfu - this->min_intensity)
        / (this->max_intensity - this->min_intensity) * 255 );

    if (scaled > 255) {
        return 255;
    } else if (scaled < 0) {
        return 0;
    }

    return scaled;
}

void
PortalWidget::li_clamp_2d (
    int *ij_f,    /* Output: "floor" pixel in moving img in vox*/
    int *ij_r,    /* Ouptut: "round" pixel in moving img in vox*/
    float *li_1,  /* Output: Fraction for upper index voxel */
    float *li_2,  /* Output: Fraction for lower index voxel */
    float *ij     /* Input:  Unrounded pixel coordinates in vox */
)
{
    for (int n=0; n<2; n++) {
        float maff = floor (ij[n]);
        ij_f[n] = (int) maff;
        ij_r[n] = ROUND_INT (ij[n]);
        li_2[n] = ij[n] - maff;
        if (ij_f[n] < 0) {
            ij_f[n] = 0;
            ij_r[n] = 0;
            li_2[n] = 0.0f;
        } else if (ij_f[n] >= this->ijk_max[n]-1) {
            ij_f[n] = this->ijk_max[n] - 2;
            ij_r[n] = this->ijk_max[n] - 1;
            li_2[n] = 1.0f;
        }
        li_1[n] = 1.0f - li_2[n];
    }
}

void
PortalWidget::renderSlice (int slice_num)
{
    int i, j;                /* portal coords        */
    float xy[2];             /* real space coords    */
    float ij[2];             /* volume slice indices */
    int ij_f[2];
    int ij_r[2];
    float li_1[2], li_2[2];  /* linear interpolation fractions */
    float hfu;
    int idx, shade;
    float contrib[4];
    QColor pixColor;
    QPainter painter (this->slice);

    float* img = (float*) (this->vol)->img;

    /* Set slice pixels */
    for (j=0; j<this->dim[1]; j++) {
        xy[1] = this->res[1]*(float)j;
        ij[1] = xy[1] / this->spacing[1];
        for (i=0; i<this->dim[0]; i++) {
            xy[0] = this->res[0]*(float)i;
            ij[0] = xy[0] / this->spacing[0];

            this->li_clamp_2d (ij_f, ij_r, li_1, li_2, ij);

            idx = this->stride[2] * slice_num
                  + this->stride[1] * ij_f[1]
                  + this->stride[0] * ij_f[0];
        	contrib[0] = li_1[0] * li_1[1] * img[idx];

            idx = this->stride[2] * slice_num
                  + this->stride[1] * ij_f[1]
                  + this->stride[0] * (ij_f[0]+1);
        	contrib[1] = li_2[0] * li_1[1] * img[idx];

            idx = this->stride[2] * slice_num
                  + this->stride[1] * (ij_f[1]+1)
                  + this->stride[0] * ij_f[0];
        	contrib[2] = li_1[0] * li_2[1] * img[idx];

            idx = this->stride[2] * slice_num
                  + this->stride[1] * (ij_f[1]+1)
                  + this->stride[0] * (ij_f[0]+1);
        	contrib[3] = li_2[0] * li_2[1] * img[idx];

            hfu = contrib[0] + contrib[1] + contrib[2] + contrib[3];
            shade = this->getPixelValue (hfu);
            pixColor.setRgb (shade, shade, shade);
            painter.setPen (pixColor);
            painter.drawPoint (i, j);
        }
    }

    /* Deal with inverted y-axis */
    if ((this->view == PV_CORONAL) || (this->view == PV_SAGITTAL)) {
        QImage tmp = this->slice->mirrored();
        this->pmap = QPixmap::fromImage (tmp);
    } else {
        this->pmap = QPixmap::fromImage (*(this->slice));
    }

    (this->pmi)->setPixmap (this->pmap);

    this->current_slice = slice_num;
}

void
PortalWidget::setVolume (Volume* vol)
{
    this->vol = vol;
    float* img = (float*) vol->img;
    
    /* Obtain value range */
    this->min_intensity = FLT_MAX;
    this->max_intensity = FLT_MIN;
    for (int i=0; i<vol->npix; i++) {
        if ( img[i] < this->min_intensity ) {
            this->min_intensity = img[i];
        }
        if ( img[i] > this->max_intensity ) {
            this->max_intensity = img[i];
        }
    }

    /* Display the loaded volume */
    PortalWidget::setView (PV_AXIAL);
    PortalWidget::renderSlice (0);
}

void
PortalWidget::doZoom (int step)
{
    if ( (this->current_slice + step < this->ijk_max[2]) &&
         (this->current_slice + step >= 0)  ) {
        this->current_slice += step;
        this->renderSlice(this->current_slice);
    }
}

void
PortalWidget::wheelEvent (QWheelEvent *event)
{
    /* Note: delta is in "# of 8ths of a degree"
     *   Each wheel click is usually 15 degrees on most mice */
    int step = event->delta() / (8*15);

    this->doZoom (step);
}

void
PortalWidget::keyPressEvent (QKeyEvent *event)
{
    switch (event->key())
    {
    case Qt::Key_1:
        this->setView (PV_AXIAL);
        break;
    case Qt::Key_2:
        this->setView (PV_CORONAL);
        break;
    case Qt::Key_3:
        this->setView (PV_SAGITTAL);
        break;
    case Qt::Key_Minus:
        this->doZoom (-1);
        break;
    case Qt::Key_Plus:
    case Qt::Key_Equal:
        this->doZoom (1);
        break;
    default:
        /* Forward to default callback */
        QGraphicsView::keyPressEvent (event);
    }
}

void
PortalWidget::mousePressEvent (QMouseEvent *event)
{
    int i,j;
    float xy[2];

    i = event->pos().x();
    j = event->pos().y();

    xy[0] = (float)i*this->res[0];
    xy[1] = (float)j*this->res[1];

    printf ("   Portal Coords: %i %i\n", i, j);
    printf ("RealSpace Coords: %f %f\n", xy[0], xy[1]);
    printf ("    Slice Coords: %f %f\n", xy[0] / this->spacing[0],
                                         xy[1] / this->spacing[1]);
}
