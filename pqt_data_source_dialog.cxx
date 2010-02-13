/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <QtGui>
#include "pqt_data_source_dialog.h"
#include "pqt_database.h"

Pqt_data_source_dialog::Pqt_data_source_dialog ()
{
    setupUi (this); // this sets up the GUI

    /* Hide status */
    this->label_status->hide();

    /* Attach model to QT listView */
    this->m_data_source_list_model = new Pqt_data_source_list_model;
    this->listView_data_source_list->setModel (this->m_data_source_list_model);
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
    /* Validate input */
    if (this->lineEdit_data_source_name->text().isEmpty()) {
	QMessageBox::information (0, QString ("Info"), 
	    QString ("Please fill in the data source name."));
	return;
    }
    if (this->lineEdit_host->text().isEmpty()) {
	QMessageBox::information (0, QString ("Info"), 
	    QString ("Please fill in the hostname."));
	return;
    }
    if (this->lineEdit_port->text().isEmpty()) {
	QMessageBox::information (0, QString ("Info"), 
	    QString ("Please fill in the port."));
	return;
    }
    if (this->lineEdit_aet->text().isEmpty()) {
	QMessageBox::information (0, QString ("Info"), 
	    QString ("Please fill in AET."));
	return;
    }

    /* Insert into database */
    pqt_database_insert_data_source (
	this->lineEdit_data_source_name->text(),
	this->lineEdit_host->text(),
	this->lineEdit_port->text(),
	this->lineEdit_aet->text());

    /* Refresh model */
    this->m_data_source_list_model->load_query ();
}

void
Pqt_data_source_dialog::pushbutton_delete_released (void)
{
    QMessageBox::information (0, QString ("Info"), 
	QString ("Pushed delete"));
}
