/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dips_panel_h_
#define _dips_panel_h_

class Dips_panel {
public:
    Dips_panel ();
    ~Dips_panel ();
    void open_panel (int panel_no, int height, int width);
    void poll_dummy (void);

public:
    int panel_no;
    int height, width;     /* width == x, height == y */
    struct PANEL *panelp;
    unsigned short* pixelp;
};

#endif
