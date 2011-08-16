#ifndef _cview_main_h_
#define _cview_main_h_

#include <QtGui>
#include <QApplication>
#include <QMenu>
#include "volume.h"
#include "cview_portal.h"

class PortalGrid : public QWidget
{
    public:
        PortalGrid (Volume* input_vol, QWidget* parent = 0);

    public:
        PortalWidget* portal0;
        PortalWidget* portal1;
        PortalWidget* portal2;
        PortalWidget* portal3;
};

class CrystalWindow : public QMainWindow
{
    Q_OBJECT;

    public:
        CrystalWindow (int argc, char** argv, QWidget *parent = 0);

    public slots:
        void open ();

    private: /* methods */
        bool openVol (const char* fn);

    private: /* variables */
        QMenu *menuFile;
        QAction *itemOpen;
        QAction *itemExit;
        PortalGrid *portalGrid;
        Volume* input_vol;
};


#endif
