/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include "frame.h"
#include "ise_globals.h"
#include "synthetic_source.h"
#include "synthetic_source_thread.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Synthetic_source::Synthetic_source ()
{
    this->thread = new Synthetic_source_thread;
    this->thread->set_synthetic_source (this);
    this->thread->start ();
}

unsigned long
Synthetic_source::get_size_x () {
    return 2048;
}

unsigned long
Synthetic_source::get_size_y () {
    return 1536;
}

const std::string 
Synthetic_source::get_type () {
    return std::string("Synthetic");
}

/* -------------------------------------------------------------------------*
   Simulated-related routines
 * -------------------------------------------------------------------------*/
void
simulate_image_ramp (Frame* f, int x_size, int y_size)
{
    int x, y;
    unsigned short* p;

    /* Fill in dummy image */
    p = (unsigned short*) f->img;
    for (y = 0; y < y_size; y++) {
	for (x = 0; x < x_size; x++) {
	    if (y < 50 || y > y_size - 50) {
		*p = MAXGREY;
	    } else {
		*p = (unsigned short) (MAXGREY * ((double) x / (double) x_size));
	    }
	    p++;
	}
    }
}

static void
simulate_clip_pos_sin (Frame* f, int x_size, int y_size)
{
    static double phase = 0.0;
    double amp = 50.0;

    int xpos = x_size / 2;
    int ypos = (y_size / 2) + (int) (amp*sin(phase));
    f->clip_x = xpos;
    f->clip_y = ypos;
    // 15fps=60f/cyc
    phase += (M_PI / 30.0);	/* 60 frames per cycle */
}

static void
simulate_image_fill_bg (Frame* f, int x_size, int y_size, unsigned short bg_color)
{
    int i;
    unsigned long longfill = (bg_color << 16) | bg_color;
    int reps = x_size * y_size / 2;  /* sizes are always even */
    unsigned long* longbuf = (unsigned long*) f->img;

    for (i = 0; i < reps; i++) {
	longbuf[i] = longfill;
    }
}

static void
simulate_image_fill_fg (Frame* f, int x_size, int y_size, unsigned short fg_color)
{
    int xmin = f->clip_x - 3, xmax = f->clip_x + 3;
    int ymin = f->clip_y - 5, ymax = f->clip_y + 5;
    int x, y;

    for (y = ymin; y <= ymax; y++) {
	for (x = xmin; x <= xmax; x++) {
	    f->img[y*x_size+x] = fg_color;
	}
    }

}

void
simulate_image_clip_sin (Frame* f, int x_size, int y_size)
{
    simulate_image_fill_bg (f, x_size, y_size, 800);
    simulate_clip_pos_sin (f, x_size, y_size);
    simulate_image_fill_fg (f, x_size, y_size, 200);
}

void
simulate_image_clip_fluoro_pulse (Frame* f, int x_size, int y_size)
{
    static int countdown1 = 10;
    static int countdown2 = 20;
    if (countdown1 > 0 || countdown2 <= 0) {
	if (--countdown1 < 0) countdown1 = 0;
	simulate_image_fill_bg (f, x_size, y_size, 30);
	simulate_clip_pos_sin (f, x_size, y_size);
	simulate_image_fill_fg (f, x_size, y_size, 10);
    } else {
	--countdown2;
	simulate_image_fill_bg (f, x_size, y_size, 300);
	simulate_clip_pos_sin (f, x_size, y_size);
	simulate_image_fill_fg (f, x_size, y_size, 100);
    }
}


void
simulate_image_clip_fluoro_pulses (Frame* f, int x_size, int y_size)
{
    static int countdown1 = 10;
    static int countdown2 = 10;
    if (countdown1 > 0) {
	if (--countdown1 == 0) {
	    countdown2 = 10;
	}
	simulate_image_fill_bg (f, x_size, y_size, 20);
	simulate_clip_pos_sin (f, x_size, y_size);
	simulate_image_fill_fg (f, x_size, y_size, 10);
    } else {
	if (--countdown2 == 0) {
	    countdown1 = 10;
	}
	simulate_image_fill_bg (f, x_size, y_size, 300);
	simulate_clip_pos_sin (f, x_size, y_size);
	simulate_image_fill_fg (f, x_size, y_size, 100);
    }
}

void
simulate_image_clip_line (Frame* f, int x_size, int y_size)
{
    int y;
    simulate_image_fill_bg (f, x_size, y_size, 800);
    simulate_clip_pos_sin (f, x_size, y_size);
    for (y = 0; y < y_size; y++) {
	int x1 = y;
	int x2 = y + (x_size-y_size);
	int x3 = x_size - x1 - 1;
	int x4 = x_size - x2 - 1;
	f->img[y*x_size+x1] = 400;
	f->img[y*x_size+x2] = 400;
	f->img[y*x_size+x3] = 400;
	f->img[y*x_size+x4] = 400;
    }
    simulate_image_fill_fg (f, x_size, y_size, 200);
}

static void
simulate_image (Frame* f, int x_size, int y_size)
{
    simulate_image_clip_sin (f, x_size, y_size);
#if defined (commentout)
    simulate_image_ramp (f, x_size, y_size);
    simulate_image_clip_line (f, x_size, y_size);
    simulate_image_clip_fluoro_pulse (f, x_size, y_size);
    simulate_image_clip_fluoro_pulses (f, x_size, y_size);
#endif
}

void
Synthetic_source::grab_image (Frame* f)
{
    simulate_image (f, 2048, 1536);
}
