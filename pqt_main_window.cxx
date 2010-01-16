/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <QtGui>
#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlError>
#include <QtSql/QSqlQuery>

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

static void
print_database_error (QSqlError sql_error)
{
    QMessageBox::information (0, QString ("Database error"),
	QString ("Database error: %1, %2, %3")
	.arg(sql_error.type())
	.arg(sql_error.number())
	.arg(sql_error.text()));
}

void
Pqt_main_window::test_database ()
{
    QSqlDatabase db = QSqlDatabase::addDatabase ("QSQLITE");
    //db.setDatabaseName (":memory:");

    /* For sqlite, QSqlDatabase::setDatabaseName is where we pass the
       name of the sqlite file. */
    db.setDatabaseName ("deleteme.sqlite");
    bool ok = db.open ();
    if (!ok) {
	print_database_error (db.lastError());
	return;
    }

    QSqlQuery query;
    QString sql = "CREATE TABLE IF NOT EXISTS patient_screenshots ( oi INTEGER PRIMARY KEY, patient_id TEXT, patient_name TEXT, screenshot_timestamp DATE );";

    ok = query.exec (sql);

    if (!ok) {
	print_database_error (query.lastError());
	return;
    }

    sql = 
	"SELECT patient_id,patient_name,datetime(MAX(screenshot_timestamp)) "
	"FROM patient_screenshots GROUP BY patient_id,patient_name "
	"ORDER BY MAX(screenshot_timestamp) DESC;";
    ok = query.exec (sql);
    if (!ok) {
	print_database_error (query.lastError());
	return;
    }
    
    db.close ();
}

void
Pqt_main_window::new_data_source (void)
{
    /* Open dialog */
    m_data_source_dialog->show ();
}
