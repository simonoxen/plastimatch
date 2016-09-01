/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "register_gui_load_dialog.h"

Register_gui_load_dialog::Register_gui_load_dialog ()
{
    setupUi (this); // this sets up the GUI
}

Register_gui_load_dialog::~Register_gui_load_dialog ()
{
    
}

QString
Register_gui_load_dialog::get_fixed_pattern ()
{
    return lineEdit_fixedPattern->text ();
}
