/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _acquire_4030e_window_h_
#define _acquire_4030e_window_h_
#include "ise_config.h"
#include <QAction>
#include <QMainWindow>
#include <QMenu>
#include <QSystemTrayIcon>
#include "ui_acquire_4030e_window.h" //virtual h file

class QString;
class QSystemTrayIcon;
class QTimer;

class Acquire_4030e_window : public QMainWindow,
                             private Ui::ui_acquire_4030e_window
{
    Q_OBJECT
    ;
public:
    Acquire_4030e_window ();    
protected:
    void closeEvent(QCloseEvent *event);
public:
    enum Label_style {
	LABEL_NOT_READY,
	LABEL_ACQUIRING, //Yellow
	LABEL_PREPARING, //Orange
	LABEL_READY
    };

public:
    void create_actions ();
    void create_tray_icon ();
    //void set_icon ();
    void set_icon (int idx,Label_style style);   
    void log_output (const QString& log);
    void set_label (int panel_no, const QString& log);
    void set_label_style (int panel_no, Label_style style);

public slots:
    void request_quit ();	
    void systray_activated (QSystemTrayIcon::ActivationReason reason);
    void RestartPanel_0 ();
    void RestartPanel_1 ();    
    void ShowPanelControl_0 ();    
    void ShowPanelControl_1 ();
	void TimerReadyToQuit_event();
	void FinalQuit ();

public:
    QAction *show_action;
    QAction *quit_action;
    QSystemTrayIcon *tray_icon1;
    QSystemTrayIcon *tray_icon2;

    QMenu *tray_icon_menu;
	//void FinalQuit ();

	QTimer* m_TimerReadyToQuit;
   

    //added by YKP
    void UpdateLabel(int iPanelIdx, Label_style enStyle); // 0 based panel ID //called from child proc except the first time
    //void UpdateTray(Label_style enStyle);

	bool m_bSeqKillReady;
    
    // how to access parent window?
    //bool m_bPanel1Ready
        
};

#endif
