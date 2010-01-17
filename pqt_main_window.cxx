/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <QtGui>

#include "pqt_data_source_dialog.h"
#include "pqt_database.h"
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

    /* QT doesn't seem to have an API for getting the user's application 
       data directory.  So we construct a hypothetical ini file name, 
       then grab the directory. */
    QSettings tmp (
	QSettings::IniFormat, /* Make sure we get path, not registry */
	QSettings::UserScope, /* Get user directory, not system direcory */
	"Plastimatch",        /* Orginazation name (subfolder within path) */
	"plastimatch_qt"      /* Application name (file name with subfolder) */
    );
    QString config_dir = QFileInfo(tmp.fileName()).absolutePath();

#if defined (commentout)
    QMessageBox::information (0, QString ("Info"), 
	QString ("Config dir is %1").arg (config_dir));
#endif

    /* Construct filename of sqlite database that holds settings */
    QSettings settings;
    QString db_path = settings.value ("db/sqlite3_path", 
	QFileInfo (QDir (config_dir), QString ("pqt.sqlite"))
	.absoluteFilePath()).toString();

    /* Load database */
    pqt_database_start (db_path);
}

Pqt_main_window::~Pqt_main_window ()
{
    delete m_data_source_dialog;
    delete m_patient_list_model;

    QSettings settings;
    settings.sync ();
}

void
Pqt_main_window::new_data_source (void)
{
    /* Open dialog */
    m_data_source_dialog->show ();
}
