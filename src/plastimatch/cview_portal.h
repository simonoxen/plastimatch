#ifndef _cview_portal_h_
#define _cview_portal_h_

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QtGui/QGraphicsView>
#include "volume.h"

/* Custom Types */
enum PortalViewType {
    PV_AXIAL,
    PV_CORONAL,
    PV_SAGITTAL
};


/* Derived classes of QGraphicsView */
class PortalWidget : public QGraphicsView
{
    Q_OBJECT  /* Needed for QT signals/slots */

    public:
        PortalWidget (QWidget *parent = 0);

    signals:
        void cursorModified (float x, float y, float z);
        void sliceModified (int n);

    public slots:
        void setVolume (Volume* vol);
        void setView (enum PortalViewType view);
        void setCursor (float x, float y, float z);

    protected:
       void wheelEvent (QWheelEvent *event);
       void keyPressEvent (QKeyEvent *event);
       void mousePressEvent (QMouseEvent *event);
       void resizeEvent (QResizeEvent *event);

    private:
        void renderSlice (int slice_num);
        int getPixelValue (float hfu);
        void doZoom (int step);
        void li_clamp_2d (int *ij_f, int *ij_r, float *li_1, float *li_2, float *ij);

    private:
        /* Qt4 rendering stuff */
        QGraphicsScene* scene;      /* Portal's QGraphicsScene    */
        uchar* fb;                  /* Framebuffer for drawing    */
        QImage* slice;              /* QImage tied to Framebuffer */
        QPixmap pmap;               /* QPixmap for rendering      */
        QGraphicsPixmapItem* pmi;   /* Needed to update QPixmap   */

        /* PortalWidget stuff */
        Volume* vol;            /* Plastimatch volume                    */
        PortalViewType view;    /* PV_AXIAL or PV_CORONAL or PV_SAGITTAL */
        float min_intensity;    /* Min floating point value from volume  */
        float max_intensity;    /* Max floating point value from volume  */
        int current_slice;      /* Current slice index within volume     */
        int ijk_max[3];         /* index limits for current view type    */
        int stride[3];          /* volume strides for current view type  */
        int dim[2];             /* portal dimensions (in pixels)         */
        float res[2];           /* portal resolution (in mm per pixel)   */
        float spacing[2];       /* voxel spacing in slice (in mm)        */
        float offset[2];        /* volume slice offset (in mm)           */
};

#endif
