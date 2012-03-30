/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _acquire_4030e_child_h_
#define _acquire_4030e_child_h_

#include <QCoreApplication>

class Dips_panel;
class Varian_4030e;

class Acquire_4030e_child : public QCoreApplication
{
    Q_OBJECT
    ;
public slots:
    void run();
    
public:
    Acquire_4030e_child (int argc, char* argv[]);
    ~Acquire_4030e_child ();
public:
    void open_receptor (const char* path);
public:
    int idx;
    Dips_panel *dp;
    Varian_4030e *vp;
};

#endif
