/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cview_portal_h_
#define _cview_portal_h_

#include <QtGui/QGraphicsView>
#include "volume.h"

class PortalWidget : public QGraphicsView
{
    Q_OBJECT  /* Needed for QT signals/slots */

    public:
        enum PortalViewType {
            Axial,
            Coronal,
            Sagittal
        };

    private:
        /* Qt4 rendering */
        QGraphicsScene* scene;      /* Portal's QGraphicsScene    */
        uchar* fb;                  /* Framebuffer for drawing    */
        QImage* slice;              /* QImage tied to Framebuffer */
        QPixmap pmap;               /* QPixmap for rendering      */
        QGraphicsPixmapItem* pmi;   /* Needed to update QPixmap   */

        /* PortalWidget */
        Volume* vol;            /* Plastimatch volume                    */
        PortalViewType view;    /* Axial or Coronal or Sagittal          */
        float min_intensity;    /* Min floating point value from volume  */
        float max_intensity;    /* Max floating point value from volume  */
        int current_slice;      /* Current slice index within volume     */
        int ijk_max[3];         /* index limits for current view type    */
        int stride[3];          /* volume strides for current view type  */
        int dim[2];             /* rendering surface dims (in pixels)    */
        float res[2];           /* portal resolution (in mm per pixel)   */
        float spacing[2];       /* voxel spacing in slice (in mm)        */
        float offset[2];        /* volume slice offset (in mm)           */
        float sfactor;          /* Scaling factor                        */

        /* Scroll/Panning */
        bool pan_mode;
        int pan_xy[2];


    public:
        PortalWidget (QWidget *parent = 0);
        int getSlices ();                        /* get # of slices in view */

    private:
        int getPixelValue (float hfu);
        void setScaleFactor ();
        void doZoom (int step);
        void doScale (float step);
        void li_clamp_2d (int *ij_f, int *ij_r, float *li_1, float *li_2, float *ij);

    signals:
        void targetChanged (float* xyz);                /* on cursor change */
        void sliceChanged (int n);                      /* on slice change  */

    public slots:
        void setVolume (Volume* vol);               /* Attach volume to portal  */
        void setView (enum PortalViewType view);    /* Set portal view type     */
        void setTarget (float* xyz);                /* Set the cursor RS coords */
        void renderSlice (int slice_num);           /* Render slice_num         */

    protected:
        void wheelEvent (QWheelEvent *event);
        void keyPressEvent (QKeyEvent *event);
        void mousePressEvent (QMouseEvent *event);
        void mouseReleaseEvent (QMouseEvent *event);
        void mouseMoveEvent (QMouseEvent *event);
        void resizeEvent (QResizeEvent *event);
};

#endif
