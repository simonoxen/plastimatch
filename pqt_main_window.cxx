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
