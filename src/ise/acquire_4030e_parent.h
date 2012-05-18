/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _acquire_4030e_parent_h_
#define _acquire_4030e_parent_h_
#include "ise_config.h"
#include <QApplication>
#include <QProcess>

class Acquire_4030e_window;
class Advantech;
class QTimer;

class Acquire_4030e_parent : public QApplication
{
    Q_OBJECT
    ;
public:
    Acquire_4030e_parent (int argc, char* argv[]);
    ~Acquire_4030e_parent ();
public:
    enum Generator_state {
	WAITING,
	EXPOSE_REQUEST,
	EXPOSING
    };
public:
    void initialize (int argc, char* argv[]);
    void kill_rogue_processes ();
    void log_output (const QString& log);
public slots:
    void timer_event ();
    void poll_child_messages ();
    void about_to_quit ();
public:
    Acquire_4030e_window *window;
    int num_process;
    QProcess process[2];
    QTimer *timer;
    Advantech *advantech;
    //bool generator_prep;
    Generator_state generator_state;
    bool panel_select;
    int panel_timer;
};

#endif
