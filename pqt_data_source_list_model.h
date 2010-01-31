/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pqt_data_source_list_model_h_
#define _pqt_data_source_list_model_h_

#include "plm_config.h"
#include "ui_pqt_main_window.h"

class Pqt_data_source_list_model : public QAbstractTableModel {
    Q_OBJECT
    ;

public:
    Pqt_data_source_list_model (QObject *parent = 0)
	: QAbstractTableModel (parent) {}
    ~Pqt_data_source_list_model () {}

    int rowCount (const QModelIndex& parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;
    QVariant data (const QModelIndex& index, int role) const;
    QVariant headerData (int section, Qt::Orientation orientation,
	int role = Qt::DisplayRole) const;
};
#endif
