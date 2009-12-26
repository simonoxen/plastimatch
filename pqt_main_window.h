/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pqt_main_window_h_
#define _pqt_main_window_h_

#include "plm_config.h"
#include <QMainWindow>

QT_BEGIN_NAMESPACE
// class QAction;
// class QDialogButtonBox;
// class QGroupBox;
// class QLabel;
// class QLineEdit;
class QMenu;
// class QMenuBar;
// class QPushButton;
// class QTextEdit;
QT_END_NAMESPACE

class Pqt_main_window : public QMainWindow {
    Q_OBJECT
    ;

public:
    Pqt_main_window ();

private:
    QMenu *file_menu;
};
#endif
