/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pqt_main_window_h_
#define _pqt_main_window_h_

#include "plm_config.h"
#include <QDialog>

QT_BEGIN_NAMESPACE
class QAction;
class QDialogButtonBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QMenu;
class QMenuBar;
class QPushButton;
class QTextEdit;
QT_END_NAMESPACE

class Pqt_main_window : public QDialog {
    Q_OBJECT
    ;

public:
    Pqt_main_window ();

private:
    QMenuBar *menuBar;
};
#endif
