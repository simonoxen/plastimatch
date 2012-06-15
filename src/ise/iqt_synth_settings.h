/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _iqt_synth_settings_h_
#define _iqt_synth_settings_h_

//#include "plm_config.h"
//#include "iqt_data_source_dialog.h"
//#include "iqt_findscu.h"
//#include "iqt_patient_list_model.h"
#include "ui_iqt_synth_settings.h"

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

class Iqt_synth_settings : public QDialog, private Ui::iqtSynthSettings {
    Q_OBJECT
    ;

public:
    Iqt_synth_settings ();
    ~Iqt_synth_settings ();

//    void render_sphere ();

    //Iqt_data_source_dialog *m_data_source_dialog;

    //Iqt_patient_list_model *m_patient_list_model;

    //Iqt_findscu m_findscu;
    int set1;
    int set2;
    int set3;
    int set4;
    int set5;

public slots:
    void slot_proceed (void);
    void slot_cancel (void);
    //void slot_settings (void);
};
#endif
