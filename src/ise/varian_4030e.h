/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _varian_4030e_h_
#define _varian_4030e_h_

class Dips_panel;

class Varian_4030e {
public:
    Varian_4030e ();
    ~Varian_4030e ();

    static const char* error_string (int error_code);

    int check_link();
    void print_sys_info ();
    int perform_gain_calibration ();
    int rad_acquisition (Dips_panel *dp);
    int perform_sw_rad_acquisition ();
    int get_image_to_file (int xSize, int ySize, 
	char *filename, int imageType=VIP_CURRENT_IMAGE);
    int get_image_to_dips (Dips_panel *dp, int xSize, int ySize);

public:
    int current_mode;
};

int CheckRecLink();
int DisableMissingCorrections (int result);

#endif
