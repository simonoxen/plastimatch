/* JAS - 2011.08.14
 *   This is CrystalView... which is currently more or less just a 
 *   testbed for my PortalWidget Qt4 class.  All of this is in
 *   very early stages of development.
 */

#include "plm_config.h"
#include <iostream>
#include <QtGui>
#include <QWidget>
#include "volume.h"
#include "mha_io.h"
#include "cview_portal.h"

using namespace std;

int
main (int argc, char** argv)
{
    Volume* input_vol;

    if (argc > 1) {
        input_vol = read_mha (argv[1]);
    } else {
        cout << "CrystalView 0.01a\n"
             << "Usage: cview mha_file\n\n"
             << "  When Running:\n"
             << "    1 - axial view\n"
             << "    2 - coronal view\n"
             << "    3 - sagittal view\n\n"
             << "  -/+ - change active slice\n"
             << "  (or mouse wheel)\n\n";

        exit (0);
    }

    volume_convert_to_float (input_vol);

    QApplication app (argc, argv);

    /* Create a 512 x 512 portal and demo it a little */
    PortalWidget portal0 (512, 512);
    portal0.setVolume (input_vol);
    portal0.setView (PV_AXIAL);
    portal0.show();

    return app.exec ();
}
