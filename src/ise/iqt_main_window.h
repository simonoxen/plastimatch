/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _iqt_main_window_h_
#define _iqt_main_window_h_

//#include "plm_config.h"
//#include "iqt_data_source_dialog.h"
//#include "iqt_findscu.h"
//#include "iqt_patient_list_model.h"
#include "ui_iqt_main_window.h"

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

class Frame;
class Tracker;

class Iqt_main_window : public QMainWindow, private Ui::iqtMainWindow {
    Q_OBJECT
    ;

public:
    Iqt_main_window ();
    ~Iqt_main_window ();

    //virtual bool event (QEvent *event);

    QTimer *m_qtimer;
    bool playing;
    bool synth;
    unsigned int width;
    unsigned int height;
    QString filename;
//    Frame* f;
    QSlider *framePos;
    int numFiles;
    Frame* frameList[150];
    Tracker *tracker;

public slots:
    void get_new_frame (int pos);
    void slot_load (void);
    void show_fluoro (QString path);
    void slot_save (void);
    void slot_synth (void);
    void slot_play_pause (void);
    void slot_stop (void);
    void slot_go_back(void);
    void slot_go_forward(void);
    void slot_timer (void);
    void slot_reload_frame (void);
    void slot_frame_ready (int width, int height);

signals:
    void fluoro_ready (QString path);
};
#endif
