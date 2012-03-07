/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>
#include <dos.h>
#include <math.h>

#include "HcpErrors.h"
#include "HcpFuncDefs.h"
#include "iostatus.h"

#include "advantech.h"
#include "dips_panel.h"
#include "varian_4030e.h"


// The string such as "A422-07" is the imager serial number
char *default_path_1 = "C:\\IMAGERs\\A422-07"; // Path to IMAGER tables
char *default_path_2 = "C:\\IMAGERs\\A663-11"; // Path to IMAGER tables

// In the following the mode number for the rad mode required should be set
//int  crntModeSelect = 0;


#define ESC_KEY   (0x1B)
#define ENTER_KEY (0x0D)

#define HCP_SIGNAL_TIMEOUT        (-2)
#define HCP_SIGNAL_KEY_PRESSED    (-1)

//----------------------------------------------------------------------
//  main
//----------------------------------------------------------------------
int 
main(int argc, char* argv[])
{
    Advantech advantech;
    char *path_1 = default_path_1;
    char *path_2 = default_path_2;
    int choice = 0;
    int result;

#define HIRES_IMAGE_HEIGHT 3200
#define HIRES_IMAGE_WIDTH 2304

    printf ("Welcome to acquire_4030e\n");

    // Check for receptor path on the command line
    if (argc > 1) {
	path_1 = argv[1];
    }
    if (argc > 2) {
	path_2 = argv[2];
    }

    Dips_panel dp;
    dp.open_panel (0, HIRES_IMAGE_HEIGHT, HIRES_IMAGE_WIDTH);

    Varian_4030e vp;
    result = vp.open_receptor_link (path_1);
    if (result != HCP_NO_ERR) {
	printf ("vp.open_receptor_link returns error %d\n", result);
        return -1;
    }

    // The following call is for test purposes only
    result = DisableMissingCorrections(result);

    result = vp.check_link ();
    if (result != HCP_NO_ERR) {
	printf ("vp.check_link returns error %d\n", result);
        vip_close_link();
        return -1;
    }

    vp.print_sys_info ();

    result = vip_select_mode (vp.current_mode);

    if (result != HCP_NO_ERR) {
        printf ("vip_select_mode(%d) returns error %d\n", 
            vp.current_mode, result);
        vip_close_link();
        return -1;
    }

    while (true) {
#if defined (commentout)
        /* Wait for generator expose request */
        while (!advantech.ready_for_expose ()) {
            Sleep (10);
        }
#endif
        /* Get frame from panel */
        printf ("Waiting for image.\n");
        vp.rad_acquisition (&dp);
    }

    vip_close_link();

    return 0;
}
