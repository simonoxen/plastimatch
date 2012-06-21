/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <QDebug>

#include "cbuf.h"
#include "frame.h"
#include "ise_globals.h"
#include "iqt_main_window.h"
#include "sleeper.h"
#include "synthetic_source.h"
#include "synthetic_source_thread.h"

Synthetic_source_thread::Synthetic_source_thread (int width, int height, int fps) 
{
    this->width = width;
    this->height = height;
    this->sleep = 1000/fps;
}

Synthetic_source_thread::~Synthetic_source_thread () {
}

void 
Synthetic_source_thread::set_synthetic_source (Synthetic_source *ss)
{
    this->ss = ss;
}

void
Synthetic_source_thread::run () {
    while (playing) {
        //qDebug() << "Hello world";
        Frame *f = this->ss->cbuf->get_frame ();
        qDebug() << "Got frame.";
        Sleeper::msleep (sleep);
        
    /***********************************************************\
    |      5  fps (msleep-200) -> jumpy, 26%  CPU usage         |
    |      10 fps (msleep-100) -> fair,  46%  CPU usage         |
    |  >>> 15 fps (msleep-75)  -> fair,  62%  CPU usage <<<     |
    |      20 fps (msleep-50)  -> good,  92%  CPU usage         |
    |      25 fps (msleep-40)  -> lags,  102% CPU usage         |
    |      30 fps (msleep-33)  -> lags,  103% CPU usage *CRASH* | 
    |            <<<FULL RESOLUTION (2048 x 1536)>>>            |
    \***********************************************************/
        
        qDebug() << "Grabbing synth image.";
        this->ss->grab_image (f);
        this->ss->cbuf->add_waiting_frame (f);

        /* Send signal to main window (or widget) that frame is ready 
           The main window can call cbuf->display_lock_newest_frame ()
           to get the frame */
        //Iqt_main_window::frame_ready (f);

        emit frame_ready (f, width, height);

        qDebug() << "Done.";
    }
}

/*
void
Synthetic_source_thread::stop ()
{
    this->playing = false;
}*/
