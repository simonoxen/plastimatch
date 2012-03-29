/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _varian_4030e_h_
#define _varian_4030e_h_

#include "ise_config.h"
#include <windows.h>
#include <QMutex>
#include "HcpErrors.h"
#include "HcpFuncDefs.h"
#include "iostatus.h"

class Dips_panel;

class Varian_4030e {
public:
    Varian_4030e (int idx);
    ~Varian_4030e ();

    static QMutex vip_mutex;
    static const char* error_string (int error_code);

    int open_link (const char *path);
    int check_link();
    void close_link ();
    void print_sys_info ();
    int get_mode_info (SModeInfo &modeInfo, int current_mode);
    int print_mode_info ();
    int query_prog_info (UQueryProgInfo &crntStatus, bool showAll = FALSE);
    int wait_on_complete (UQueryProgInfo &crntStatus, int timeoutMsec = 0);
    int wait_on_num_frames (
        UQueryProgInfo &crntStatus, int numRequested, int timeoutMsec = 0);
    int wait_on_num_pulses (UQueryProgInfo &crntStatus, int timeoutMsec = 0);
    int wait_on_ready_for_pulse (UQueryProgInfo &crntStatus, 
        int timeoutMsec = 0, int expectedState = TRUE);
    int rad_acquisition (Dips_panel *dp);
    int perform_sw_rad_acquisition ();
    int get_image_to_file (int xSize, int ySize, 
	char *filename, int imageType = VIP_CURRENT_IMAGE);
    int get_image_to_dips (Dips_panel *dp, int xSize, int ySize);
    int disable_missing_corrections (int result);

public:
    int idx;
    int current_mode;
    int receptor_no;
};

int CheckRecLink();

#endif
