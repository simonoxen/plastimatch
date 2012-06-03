/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <QtGui>

#include "iqt_application.h"

Iqt_application::Iqt_application (int argc, char* argv[])
    : QApplication (argc, argv)
{
    this->foo = 3;
}

Iqt_application::~Iqt_application ()
{
}

int 
Iqt_application::get_foo (void) 
{
    return this->foo;
}
