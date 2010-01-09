/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <QtGui>
#include "pqt_main_window.h"

Pqt_main_window::Pqt_main_window ()
{
    setupUi (this); // this sets up the GUI

    m_patient_list_model = new Pqt_patient_list_model;

    /* Attach model to QT table */
    tableView->setModel (m_patient_list_model);
}

Pqt_main_window::~Pqt_main_window ()
{
    delete m_patient_list_model;
}
