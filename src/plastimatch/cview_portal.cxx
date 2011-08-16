#include <iostream>
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

// TODO: * Maintain aspect ratio within portal
//       * Pan / Zoom
//       * Implement slots & signals (fully)
//       * Qt Designer hooks

#define ROUND_INT(x) ((x)>=0?(long)((x)+0.5):(long)(-(-(x)+0.5)))


PortalWidget::PortalWidget (QWidget *parent)
    : QGraphicsView (parent)
{
    vol = NULL;
    slice = NULL;
    view = PV_AXIAL;

    /* NOTE: for some reason PortalWidget is not being bound by QGridLayout...
     *       well, not the way I think it should be, at least */
    setHorizontalScrollBarPolicy (Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy (Qt::ScrollBarAlwaysOff);
    setSizePolicy (QSizePolicy::Expanding, QSizePolicy::Expanding);

    dim[0] = size().width();
    dim[1] = size().height();

    scene = new QGraphicsScene (this);
    scene->setItemIndexMethod (QGraphicsScene::NoIndex);
    scene->setSceneRect (0, 0, dim[0], dim[1]);
    scene->setBackgroundBrush (QBrush (Qt::black, Qt::SolidPattern));
//    scene->setBackgroundBrush (QBrush (Qt::red, Qt::SolidPattern));

    setScene (scene);
}

void
PortalWidget::setView (enum PortalViewType view)
{
    this->view = view;

    /* Cannot setView with no volume attached to portal */
    if (!vol) {
        return;
    }

    /* Delete the old rendering surface */
    if (slice) {
        scene->removeItem (pmi);
        delete slice;
        free (fb);
    }

    /* Change coordinate systems */
    switch (view) {
    case PV_AXIAL:
        ijk_max[0] = vol->dim[0];
        ijk_max[1] = vol->dim[1];
        ijk_max[2] = vol->dim[2];

        stride[0] = 1;
        stride[1] = vol->dim[0];
        stride[2] = vol->dim[1] * vol->dim[0];

        spacing[0] = vol->spacing[0];
        spacing[1] = vol->spacing[1];

        offset[0] = vol->offset[0];
        offset[1] = vol->offset[1];
        break;
    case PV_CORONAL:
        ijk_max[0] = vol->dim[0];
        ijk_max[1] = vol->dim[2];
        ijk_max[2] = vol->dim[1];

        stride[0] = 1;
        stride[1] = vol->dim[1] * vol->dim[0];
        stride[2] = vol->dim[0];

        spacing[0] = vol->spacing[0];
        spacing[1] = vol->spacing[2];

        offset[0] = vol->offset[0];
        offset[1] = vol->offset[2];
        break;
    case PV_SAGITTAL:
        ijk_max[0] = vol->dim[1];
        ijk_max[1] = vol->dim[2];
        ijk_max[2] = vol->dim[0];

        stride[0] = vol->dim[0];
        stride[1] = vol->dim[1] * vol->dim[0];
        stride[2] = 1;

        spacing[0] = vol->spacing[1];
        spacing[1] = vol->spacing[2];

        offset[0] = vol->offset[1];
        offset[1] = vol->offset[2];
        break;
    default:
        exit (-1);
        break;
    }

    /* Portal resolution (mm per pix) */
    res[0] = (spacing[0] * (float)ijk_max[0]) / (float)dim[0];
    res[1] = (spacing[1] * (float)ijk_max[1]) / (float)dim[1];

    /* Make a new rendering surface */
    fb = (uchar*) malloc (4*dim[0]*dim[1]*sizeof (uchar));
    slice = new QImage (fb, dim[0], dim[1], QImage::Format_ARGB32);
    pmi = (scene)->addPixmap (pmap);

    /* We always set the portal to the center slice after changing the view */
    renderSlice (ijk_max[2] / 2);
}

int
PortalWidget::getPixelValue (float hfu)
{
    int scaled = floor ((hfu - min_intensity)
        / (max_intensity - min_intensity) * 255 );

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
        } else if (ij_f[n] >= ijk_max[n]-1) {
            ij_f[n] = ijk_max[n] - 2;
            ij_r[n] = ijk_max[n] - 1;
            li_2[n] = 1.0f;
        }
        li_1[n] = 1.0f - li_2[n];
    }
}

void
PortalWidget::renderSlice (int slice_num)
{
    int i, j;                /* portal coords        */
    int p[2];                /* frame buffer coords  */
    float xy[2];             /* real space coords    */
    float ij[2];             /* volume slice indices */
    int ij_f[2];
    int ij_r[2];
    float li_1[2], li_2[2];  /* linear interpolation fractions */
    float hfu;
    int idx, shade;

    /* Do not render if no volume is attached to portal */
    if (!vol) {
        return;
    }

    float* img = (float*) vol->img;
    uchar* pixel;

    /* Set slice pixels */
    for (j=0; j<dim[1]; j++) {
        xy[1] = res[1]*(float)j;
        ij[1] = xy[1] / spacing[1];
        for (i=0; i<dim[0]; i++) {
            xy[0] = res[0]*(float)i;
            ij[0] = xy[0] / spacing[0];

            /* Deal with Qt's inverted y-axis... this is a hack */
            if ((view == PV_CORONAL) || (view == PV_SAGITTAL)) {
                p[0] = i;  p[1] = dim[1] - j - 1;
            } else {
                p[0] = i;  p[1] = j;
            }

            this->li_clamp_2d (ij_f, ij_r, li_1, li_2, ij);

            idx = stride[2]*slice_num
                + stride[1]*(ij_f[1]+0)
                + stride[0]*(ij_f[0]+0);
        	hfu = li_1[0] * li_1[1] * img[idx];

            idx = stride[2]*slice_num
                + stride[1]*(ij_f[1]+0)
                + stride[0]*(ij_f[0]+1);
        	hfu += li_2[0] * li_1[1] * img[idx];

            idx = stride[2]*slice_num
                + stride[1]*(ij_f[1]+1)
                + stride[0]*(ij_f[0]+0);
        	hfu += li_1[0] * li_2[1] * img[idx];

            idx = stride[2]*slice_num
                + stride[1]*(ij_f[1]+1)
                + stride[0]*(ij_f[0]+1);
        	hfu += li_2[0] * li_2[1] * img[idx];

            shade = getPixelValue (hfu);
            pixel = &fb[4*dim[0]*p[1]+(4*p[0])];
            pixel[0] = (uchar)shade;    // BLUE
            pixel[1] = (uchar)shade;    // GREEN
            pixel[2] = (uchar)shade;    // RED
            pixel[3] = 0xFF;            // ALPHA
        }
    }

    /* Have Qt actually render the frame */
    pmap = QPixmap::fromImage (*slice);
    pmi->setPixmap (pmap);

    current_slice = slice_num;
    emit sliceModified (current_slice);
}

void
PortalWidget::setVolume (Volume* vol)
{
    this->vol = vol;
    float* img = (float*) vol->img;
    
    /* Obtain value range */
    min_intensity = FLT_MAX;
    max_intensity = FLT_MIN;
    for (int i=0; i<vol->npix; i++) {
        if ( img[i] < min_intensity ) {
            min_intensity = img[i];
        }
        if ( img[i] > max_intensity ) {
            max_intensity = img[i];
        }
    }

    /* Display the loaded volume */
    PortalWidget::setView (PV_AXIAL);
    PortalWidget::renderSlice (0);
}

void
PortalWidget::doZoom (int step)
{
    if ( (current_slice + step < ijk_max[2]) &&
         (current_slice + step >= 0)  ) {
        current_slice += step;
        renderSlice(current_slice);
    }
}

void
PortalWidget::wheelEvent (QWheelEvent *event)
{
    /* Note: delta is in "# of 8ths of a degree"
     *   Each wheel click is usually 15 degrees on most mice */
    int step = event->delta() / (8*15);

    doZoom (step);
}

void
PortalWidget::keyPressEvent (QKeyEvent *event)
{
    switch (event->key())
    {
    case Qt::Key_1:
        setView (PV_AXIAL);
        break;
    case Qt::Key_2:
        setView (PV_CORONAL);
        break;
    case Qt::Key_3:
        setView (PV_SAGITTAL);
        break;
    case Qt::Key_Minus:
        doZoom (-1);
        break;
    case Qt::Key_Plus:
    case Qt::Key_Equal:
        doZoom (1);
        break;
    default:
        /* Forward to default callback */
        QGraphicsView::keyPressEvent (event);
    }
}

void
PortalWidget::resizeEvent (QResizeEvent *event)
{
    /* Delete the old rendering surface */
    if (slice) {
        scene->removeItem (pmi);
        delete slice;
        free (fb);
    }

    dim[0] = event->size().height();
    dim[1] = event->size().width();

    /* Portal resolution (mm per pix) */
    res[0] = (spacing[0] * (float)ijk_max[0]) / (float)dim[0];
    res[1] = (spacing[1] * (float)ijk_max[1]) / (float)dim[1];

    /* Make a new rendering surface */
    scene->setSceneRect (0, 0, dim[0], dim[1]);
    fb = (uchar*) malloc (4*dim[0]*dim[1]*sizeof (uchar));
    slice = new QImage (fb, dim[0], dim[1], QImage::Format_ARGB32);
    pmi = (scene)->addPixmap (pmap);

    renderSlice (current_slice);
}

void
PortalWidget::setCursor (float x, float y, float z)
{
    // set realspace coords
}

void
PortalWidget::mousePressEvent (QMouseEvent *event)
{
    int i,j;
    float xy[2];

    i = event->pos().x();
    j = event->pos().y();

    xy[0] = (float)i*res[0];
    xy[1] = (float)j*res[1];

//    emit cursorModified (xyz[0], xyz[1], xyz[2]);

    /* Debug */
    std::cout << "   Portal: " << i << "  "<< j << "\n"
              << "RealSpace: " << xy[0] << "  " << xy[1] << "\n"
              << "    Slice: " << xy[0] / spacing[0] << "  "
                               << xy[1] / spacing[1] << "\n";
}
