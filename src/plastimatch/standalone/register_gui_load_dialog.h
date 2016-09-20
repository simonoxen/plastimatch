/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _register_gui_load_dialog_h_
#define _register_gui_load_dialog_h_

#include <QDialog>
#include "ui_register_gui_load_dialog.h"

class Register_gui_load_dialog
    : public QDialog, private Ui::Register_gui_load_dialog
{
    Q_OBJECT
    ;

public:
    Register_gui_load_dialog ();
    ~Register_gui_load_dialog ();
    QString get_fixed_pattern ();
};

#endif
