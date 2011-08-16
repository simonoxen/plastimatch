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

/* lazy layout */
#define PSIZE 256
#define GSIZE 10


void
CrystalWindow::openFile ()
{
     QString fileName =
         QFileDialog::getOpenFileName(this, tr("Open File"), "",
                                      tr("Images (*.mha"));

    /* Convert from QString to char array */
    QByteArray ba = fileName.toLocal8Bit();
    const char *fn = ba.data();

    /* If user opened a file, release of volume and setup new one */
    if (strcmp (fn, "")) {
        if (input_vol) {
            delete input_vol;
        }
        input_vol = read_mha (fn);
        volume_convert_to_float (input_vol);
        portal0->setVolume (input_vol);
        portal1->setVolume (input_vol);
        portal2->setVolume (input_vol);
        portal3->setVolume (input_vol);

        portal0->setView (PV_AXIAL);
        portal1->setView (PV_CORONAL);
        portal2->setView (PV_SAGITTAL);
        portal3->setView (PV_AXIAL);
    }
}


CrystalWindow::CrystalWindow (int argc, char** argv, QWidget *parent)
    :QMainWindow (parent)
{
    input_vol = read_mha (argv[1]);
    if (!input_vol) { exit (0); }
    volume_convert_to_float (input_vol);

    /* Use a static layout for now */
    this->setFixedSize (PSIZE*2+GSIZE, PSIZE*2+GSIZE+25);

    /* Add a menu */
    itemOpen = new QAction ("O&pen", this);
    itemExit = new QAction ("E&xit", this);

    menuFile = menuBar()->addMenu ("&File");
    menuFile->addAction (itemOpen);
    menuFile->addAction (itemExit);

    /* Make the menu actually do stuff */
    connect (itemExit, SIGNAL(triggered()), qApp, SLOT(quit()));
    connect (itemOpen, SIGNAL(triggered()), this, SLOT(openFile()));
    

    /* Create some portals */
    portal0 = new PortalWidget (PSIZE, PSIZE, this);
    portal0->setVolume (input_vol);
    portal0->setView (PV_AXIAL);

    portal1 = new PortalWidget (PSIZE, PSIZE, this);
    portal1->setVolume (input_vol);
    portal1->setView (PV_CORONAL);

    portal2 = new PortalWidget (PSIZE, PSIZE, this);
    portal2->setVolume (input_vol);
    portal2->setView (PV_SAGITTAL);

    portal3 = new PortalWidget (PSIZE, PSIZE, this);
    portal3->setVolume (input_vol);
    portal3->setView (PV_AXIAL);

    portal0->move (          0, 25      );
    portal1->move (PSIZE+GSIZE, 25      );
    portal2->move (          0, 25+PSIZE+GSIZE);
    portal3->move (PSIZE+GSIZE, 25+PSIZE+GSIZE);

    portal0->show();
    portal1->show();
    portal2->show();
    portal3->show();
}


int
main (int argc, char** argv)
{
    if (argc < 2 ) {
        std::cout << "CrystalView 0.02a\n"
                  << "Usage: cview mha_file\n\n"
                  << "  When Running:\n"
                  << "    1 - axial view\n"
                  << "    2 - coronal view\n"
                  << "    3 - sagittal view\n\n"
                  << "  -/+ - change active slice\n"
                  << "  (or mouse wheel)\n\n";

        exit (0);
    }

    QApplication app (argc, argv);

    CrystalWindow cview_window (argc, argv);
    cview_window.setWindowTitle ("CrystalView 0.02a");
    cview_window.show ();

    return app.exec ();
}
