/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pqt_data_source_list_model_h_
#define _pqt_data_source_list_model_h_

#include "plm_config.h"
#include <QSqlQuery>
#include "ui_pqt_main_window.h"

class Pqt_data_source_list_model : public QAbstractListModel {
    Q_OBJECT
    ;

public:
    Pqt_data_source_list_model (QObject *parent = 0)
	: QAbstractListModel (parent) { load_query (); }
    ~Pqt_data_source_list_model () {}

    /* Overrides from base class */
    int rowCount (const QModelIndex& parent = QModelIndex()) const;
    QVariant data (const QModelIndex& index, int role) const;

    /* Other methods */
    void load_query ();

public:
    mutable QSqlQuery m_query;
    int m_num_rows;
};

#endif
