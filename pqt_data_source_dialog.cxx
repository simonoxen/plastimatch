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

    /* Load data sources from database */
    QSqlQuery query = pqt_database_query_data_source_label ();
    while (query.next()) {
	QString label = query.value(0).toString();
	//this->listView_data_source_list->insert
    }
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
