#ifndef _cview_main_h_
#define _cview_main_h_

#include <QtGui>
#include <QApplication>
#include <QMenu>
#include "volume.h"
#include "cview_portal.h"

class CrystalWindow : public QMainWindow
{
    Q_OBJECT;

    public:
        CrystalWindow (int argc, char** argv, QWidget *parent = 0);

    public slots:
        void openFile ();

    private:
        QMenu *menuFile;
        QAction *itemOpen;
        QAction *itemExit;
        PortalWidget* portal0;
        PortalWidget* portal1;
        PortalWidget* portal2;
        PortalWidget* portal3;
        Volume* input_vol;
};

#endif
