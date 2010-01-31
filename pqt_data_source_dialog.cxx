/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <QtGui>
#include "pqt_data_source_dialog.h"

Pqt_data_source_dialog::Pqt_data_source_dialog ()
{
    setupUi (this); // this sets up the GUI

    foo = 3;

    this->label_status->hide();
}

Pqt_data_source_dialog::~Pqt_data_source_dialog ()
{
    
}

void
Pqt_data_source_dialog::pushbutton_new_released (void)
{
    QMessageBox::information (0, QString ("Info"), 
	QString ("Pushed new"));
}

void
Pqt_data_source_dialog::pushbutton_save_released (void)
{
    QMessageBox::information (0, QString ("Info"), 
	QString ("Pushed save: %1").arg(
	    this->lineEdit_data_source_name->text()));
}

void
Pqt_data_source_dialog::pushbutton_delete_released (void)
{
    QMessageBox::information (0, QString ("Info"), 
	QString ("Pushed delete"));
}
