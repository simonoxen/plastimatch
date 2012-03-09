/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include <QtGui>

#if (UNIX)
#include <unistd.h>
#endif

extern "C" {
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
}

#include "cview_portal.h"
#include "lua_class_image.h"
#include "lua_tty_commands_util.h"
#include "lua_tty_preview.h"
#include "plm_image.h"

/////////////////////////////////////////////////////////
// CrystalWindow : public
//

#if 0
CrystalWindow::CrystalWindow (int argc, char** argv, QWidget *parent)
    :QMainWindow (parent)
{
    input_vol = NULL;
    pli = NULL;

    createActions ();
    createMenu ();

    portalGrid = new PortalGrid;
    setCentralWidget (portalGrid);

    /* open mha from command line */
    if (argc > 1) {
        if (!openVol (argv[1])) {
            std::cout << "Failed to load: " << argv[1] << "\n";
        }
    }
}
#endif


int
preview_portal (lua_State* L, int argc, char** argv)
{
    lua_image* limg = NULL;
    Plm_image* pli;
    char* img_obj_name;

    if (argc < 2) {
        return -1;
    } else {
        img_obj_name = argv[1];
    }

    limg = (lua_image*)get_obj_ptr_from_name (L, img_obj_name);

#if 0
    printf (">>>%p\n", limg);

    if (!limg->pli) return -1;

    pid_t child_pid;
    switch (child_pid = fork()) {
        case -1:
            perror ("fork()");
            exit (1);
        case 0: 
            QApplication app (argc, argv);
            PortalWidget* portal = new PortalWidget;

            //attach pli to portal
            portal->resetPortal();
            portal->setVolume (limg->pli->gpuit_float());
            portal->setView (PortalWidget::Axial);

            exit (1);
    }
#endif

    return 0;
}
