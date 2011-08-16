/* JAS - 2011.08.14
 *   This is CrystalView... which is currently more or less just a 
 *   testbed for my PortalWidget Qt4 class.  All of this is in
 *   very early stages of development.
 */

#include "plm_config.h"
#include <iostream>
#include <QtGui>
#include <QApplication>
#include <QMenu>
#include <QMenuBar>
#include <QWidget>
#include <QFileDialog>
#include "volume.h"
#include "mha_io.h"
#include "cview_portal.h"
#include "cview_main.h"

#define VERSION "0.03a"


PortalGrid::PortalGrid (Volume* input_vol, QWidget *parent)
    :QWidget (parent)
{
    /* Create a grid layout with splitters */
    QGridLayout *grid = new QGridLayout (this);

    QSplitter *splitterT = new QSplitter (Qt::Horizontal);
    QSplitter *splitterB = new QSplitter (Qt::Horizontal);
    QSplitter *splitterV = new QSplitter (Qt::Vertical);

    /* Create some portals */
    portal0 = new PortalWidget;
    portal1 = new PortalWidget;
    portal2 = new PortalWidget;
    portal3 = new PortalWidget;

    /* place portals inside splitters */
    splitterT->addWidget (portal0);
    splitterT->addWidget (portal1);
    splitterB->addWidget (portal2);
    splitterB->addWidget (portal3);
    splitterV->addWidget (splitterT);
    splitterV->addWidget (splitterB);

    grid->addWidget (splitterV, 0, 0);
    setLayout (grid);
}


bool
CrystalWindow::openVol (const char* fn)
{
        if (input_vol) {
            delete input_vol;
        }
        input_vol = read_mha (fn);

        if (!input_vol) {
            return false;
        }

        volume_convert_to_float (input_vol);

        portalGrid->portal0->setVolume (input_vol);
        portalGrid->portal1->setVolume (input_vol);
        portalGrid->portal2->setVolume (input_vol);
        portalGrid->portal3->setVolume (input_vol);

        portalGrid->portal0->setView (PV_AXIAL);
        portalGrid->portal1->setView (PV_CORONAL);
        portalGrid->portal2->setView (PV_SAGITTAL);
        portalGrid->portal3->setView (PV_AXIAL);

        return true;
}

void
CrystalWindow::open ()
{
    QString fileName =
        QFileDialog::getOpenFileName (
                this,
                tr("Open Volume"),
                "",
                tr("MHA Volumes (*.mha)")
);

    QByteArray ba = fileName.toLocal8Bit();
    const char *fn = ba.data();

    /* Only attempt load if user selects a file */
    if (strcmp (fn, "")) {
        openVol (fn);
    }
}


CrystalWindow::CrystalWindow (int argc, char** argv, QWidget *parent)
    :QMainWindow (parent)
{
    input_vol = NULL;

    portalGrid = new PortalGrid (input_vol);
    setCentralWidget (portalGrid);

    /* Add a menu */
    itemOpen = new QAction ("&Open", this);
    itemExit = new QAction ("E&xit", this);

    menuFile = menuBar()->addMenu ("&File");
    menuFile->addAction (itemOpen);
    menuFile->addAction (itemExit);

    /* Make the menu actually do stuff */
    connect (itemExit, SIGNAL(triggered()), qApp, SLOT(quit()));
    connect (itemOpen, SIGNAL(triggered()), this, SLOT(open()));

    if (argc > 1) {
        if (!openVol (argv[1])) {
            std::cout << "Failed to load: " << argv[1] << "\n";
        }
    }
}


int
main (int argc, char** argv)
{
    std::cout << "CrystalView " << VERSION << "\n"
              << "Usage: cview [mha_file]\n\n"
              << "  When Running:\n"
              << "    1 - axial view\n"
              << "    2 - coronal view\n"
              << "    3 - sagittal view\n\n"
              << "  -/+ - change active slice\n"
              << "  (or mouse wheel)\n\n";

    QApplication app (argc, argv);

    CrystalWindow cview_window (argc, argv);
    cview_window.setWindowTitle ("CrystalView " VERSION);
    cview_window.setMinimumSize (640, 480);
    cview_window.show ();

    return app.exec ();
}
