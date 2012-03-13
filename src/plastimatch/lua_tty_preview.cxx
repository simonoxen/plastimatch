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


int preview_portal (void* pli_in)
{
#if (UNIX)
    int argc = 0;
    char** argv = NULL;
    Plm_image* pli = (Plm_image*)pli_in;

    if (!pli) return -1;

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
            portal->setVolume (pli->gpuit_float());
            portal->setView (PortalWidget::Axial);
            portal->show();
            app.exec();
            exit (0);
    }

    return 0;
#else
    fprintf (stdout, "preview not available for non-unix... yet.\n");
#endif
}
