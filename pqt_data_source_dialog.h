/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pqt_data_source_dialog_h_
#define _pqt_data_source_dialog_h_

#include "plm_config.h"
#include "ui_pqt_data_source_dialog.h"

//QT_BEGIN_NAMESPACE
// class QAction;
// class QDialogButtonBox;
// class QGroupBox;
// class QLabel;
// class QLineEdit;
// class QMenu;
// class QMenuBar;
// class QPushButton;
// class QTextEdit;
//QT_END_NAMESPACE

class Pqt_data_source_dialog : public QDialog, private Ui::pqtDataSourceDialog {
    Q_OBJECT
    ;

public:
    Pqt_data_source_dialog ();
    ~Pqt_data_source_dialog ();

    int foo;

public slots:
    void pushbutton_new_released (void);
    void pushbutton_save_released (void);
    void pushbutton_delete_released (void);

};
#endif
