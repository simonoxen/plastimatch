/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _acquire_4030e_parent_h_
#define _acquire_4030e_parent_h_
#include "ise_config.h"
#include <QApplication>
#include <QProcess>

class Acquire_4030e_window;

class Acquire_4030e_parent : public QApplication
{
    Q_OBJECT
    ;
public:
    Acquire_4030e_parent (int argc, char* argv[]);
    ~Acquire_4030e_parent ();
public:
    void initialize (int argc, char* argv[]);
    void kill_rogue_processes ();
public slots:
    void log_output ();
    void about_to_quit ();
public:
    int num_process;
    QProcess process[2];
    Acquire_4030e_window *window;
};

#endif
