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

#include "dips_panel.h"
#include "varian_4030e.h"


// The string such as "A422-07" is the imager serial number
char *default_path = "C:\\IMAGERs\\A422-07"; // Path to IMAGER tables
//char *default_path = "C:\\IMAGERs\\A663-11"; // Path to IMAGER tables

// In the following the mode number for the rad mode required should be set
//int  crntModeSelect = 0;


#define ESC_KEY   (0x1B)
#define ENTER_KEY (0x0D)

#define HCP_SIGNAL_TIMEOUT        (-2)
#define HCP_SIGNAL_KEY_PRESSED    (-1)

//----------------------------------------------------------------------
//  DisplayPrompt
//----------------------------------------------------------------------
void DisplayPrompt()
{
    printf("\n------------------------------\n");
    printf("Select operation:\n");
    printf("1 - Acquire radiographic image\n");
    printf("2 - Acquire dark image using SwHandshaking calls\n");
    printf("3 - Perform gain calibration\n");
    printf("0 - Exit\n");
}

//----------------------------------------------------------------------
//  main
//----------------------------------------------------------------------
int 
main(int argc, char* argv[])
{
    char *path = default_path;
    int choice = 0;
    int result;
    SOpenReceptorLink orl;
    Dips_panel dp;

    Varian_4030e vp;

    memset (&orl, 0, sizeof(SOpenReceptorLink));
    printf ("Welcome to acquire_4030e\n");

    // Check for receptor path on the command line
    if (argc > 1) {
	path = argv[1];
    }

    orl.StructSize = sizeof(SOpenReceptorLink);
    strcpy(orl.RecDirPath, path);

    // if we want to turn debug on so that it flushes to a file ..
    // or other settings see Virtual CP Communications Manual uncomment
    // and modify the following line if required
    //	orl.DebugMode = HCP_DBG_ON_FLSH;
    printf("Opening link to %s\n", orl.RecDirPath);
    result = vip_open_receptor_link (&orl);
    printf ("Result = %04x\n", result);

    // The following call is for test purposes only
    result = DisableMissingCorrections(result);

    printf("Calling vip_check_link\n");
    result = CheckRecLink();
    printf("vip_check_link returns %d\n", result);

    if (result == HCP_NO_ERR)
    {
	vp.print_sys_info ();

	result = vip_select_mode (vp.current_mode);

	if (result == HCP_NO_ERR)
	{
	    DisplayPrompt();
	    for (bool running = true; running;)
	    {
		if (_kbhit())
		{
		    int keyCode = _getch();
		    printf("%c\n", keyCode);
		    switch (keyCode)
		    {
		    case '1':
			vp.perform_rad_acquisition();
			break;
		    case '2':
			vp.perform_sw_rad_acquisition ();
			break;
		    case '3':
			vp.perform_gain_calibration ();
			break;
		    case '0':
			running = false;
			break;
		    }
		    if (running)
			DisplayPrompt();
		}
	    }
	}
	else {
	    printf ("vip_select_mode(%d) returns error %d\n", 
		vp.current_mode, result);
	}

	vip_close_link();
    }
    else
	printf("vip_open_receptor_link returns error %d\n", result);

    //printf("\n**Hit any key to exit");
    //_getch();
    //while(!_kbhit()) Sleep (100);

    return 0;
}
