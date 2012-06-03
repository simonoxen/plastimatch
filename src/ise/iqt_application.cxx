/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <QtGui>

#include "cbuf.h"
#include "iqt_application.h"

Iqt_application::Iqt_application (int argc, char* argv[])
    : QApplication (argc, argv)
{
    this->num_panels = 1;
    this->cbuf = new Cbuf*[this->num_panels];
    
    for (int i = 0; i < this->num_panels; i++) {
        this->cbuf[i] = new Cbuf();
    }
}

Iqt_application::~Iqt_application ()
{
    for (int i = 0; i < this->num_panels; i++) {
        delete this->cbuf[i];
    }
    delete[] this->cbuf;
}
