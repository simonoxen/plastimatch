/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _register_gui_load_dialog_h_
#define _register_gui_load_dialog_h_

#include <QDialog>
#include "register_gui.h"
#include "ui_register_gui_load_dialog.h"

class Register_gui_load_dialog
    : public QDialog, private Ui::Register_gui_load_dialog
{
    Q_OBJECT
    ;

public:
    Register_gui_load_dialog ();
    ~Register_gui_load_dialog ();
    Job_group_type get_action_pattern ();
    QString get_fixed_pattern ();
    QString get_moving_pattern ();
    bool get_repeat_for_peers ();

public slots:                
    void SLT_BrowseFixedPattern ();
    void SLT_BrowseMovingPattern ();
};

#endif
