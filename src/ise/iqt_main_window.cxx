/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <QtGui>

#include "iqt_main_window.h"

Iqt_main_window::Iqt_main_window ()
{
    /* Sets up the GUI */
    setupUi (this);

}

Iqt_main_window::~Iqt_main_window ()
{
    QSettings settings;
    settings.sync ();
}

void
Iqt_main_window::slot_load ()
{
    QMessageBox::information (0, QString ("Info"), 
	QString ("slot_load() was called"));
}
