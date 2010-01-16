/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <QtGui>
#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlError>

#include "pqt_data_source_dialog.h"
#include "pqt_main_window.h"

Pqt_main_window::Pqt_main_window ()
{
    /* Sets up the GUI */
    setupUi (this);

    /* Create data source dialog */
    m_data_source_dialog = new Pqt_data_source_dialog;

    /* Attach model to QT table in main window */
    m_patient_list_model = new Pqt_patient_list_model;
    tableView->setModel (m_patient_list_model);

    /* Set path to persistent application settings */
    QCoreApplication::setOrganizationName ("Plastimatch");
    QCoreApplication::setOrganizationDomain ("plastimatch.org");
    QCoreApplication::setApplicationName ("plastimatch_qt");

    /* Load application settings */
    QSettings settings;
    QString db_path = settings.value ("db/sqlite3_path", 
	"/tmp/pqt.sqlite").toString();

    test_database ();
}

Pqt_main_window::~Pqt_main_window ()
{
    delete m_data_source_dialog;
    delete m_patient_list_model;

    QSettings settings;
    settings.sync ();
}

void
Pqt_main_window::test_database ()
{
    QSqlDatabase db = QSqlDatabase::addDatabase ("QSQLITE");
    db.setDatabaseName (":memory:");
    bool ok = db.open();
    if (!ok) {
	QSqlError qsqlerror = db.lastError();

	QMessageBox::information (this, QString ("Database error"),
	    QString ("Error (%1,%2), %3")
	    .arg(qsqlerror.type())
	    .arg(qsqlerror.number())
	    .arg(qsqlerror.text()));
    }
}

void
Pqt_main_window::new_data_source (void)
{
    /* Open dialog */
    m_data_source_dialog->show ();
}
