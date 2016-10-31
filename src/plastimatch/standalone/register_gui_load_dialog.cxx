/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "register_gui_load_dialog.h"
#include <QFileDialog>

Register_gui_load_dialog::Register_gui_load_dialog ()
{
    setupUi (this); // this sets up the GUI
}

Register_gui_load_dialog::~Register_gui_load_dialog ()
{
    printf ("Destroying!\n");
}

Job_group_type
Register_gui_load_dialog::get_action_pattern ()
{
    if (buttonGroup_actionPattern->checkedButton() == radioButton_m2f)
    {
        return JOB_GROUP_MOVING_TO_FIXED;
    }
    else if (buttonGroup_actionPattern->checkedButton() == radioButton_a2f)
    {
        return JOB_GROUP_ALL_TO_FIXED;
    }
    else if (buttonGroup_actionPattern->checkedButton() == radioButton_a2a)
    {
        return JOB_GROUP_ALL_TO_ALL;
    }
    else
    {
        return JOB_GROUP_MOVING_TO_FIXED;
    }
}

QString
Register_gui_load_dialog::get_fixed_pattern ()
{
    return lineEdit_fixedPattern->text ();
}

QString
Register_gui_load_dialog::get_moving_pattern ()
{
    return lineEdit_movingPattern->text ();
}

bool
Register_gui_load_dialog::get_repeat_for_peers ()
{
    return checkBox_repeatForPeers->isChecked ();
}

void
Register_gui_load_dialog::SLT_BrowseFixedPattern ()
{
    QFileDialog::Options options = QFileDialog::DontResolveSymlinks;
    if (get_action_pattern() == JOB_GROUP_ALL_TO_ALL) {
        options |= QFileDialog::ShowDirsOnly;
    }
    QString pattern = QFileDialog::getExistingDirectory (this,
        tr("Open Work Directory"),
        lineEdit_fixedPattern->text (),
        options);

    lineEdit_fixedPattern->setText (pattern);
}

void
Register_gui_load_dialog::SLT_BrowseMovingPattern ()
{
    QString dirPath = QFileDialog::getExistingDirectory (this,
        tr("Open Work Directory"),
        lineEdit_movingPattern->text (),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    lineEdit_movingPattern->setText (dirPath);
}
