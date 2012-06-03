/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _iqt_application_h_
#define _iqt_application_h_

#include <QApplication>

#define ise_app ((Iqt_application *) qApp)

class Iqt_application : public QApplication {
    Q_OBJECT
    ;
public:
    Iqt_application (int argc, char* argv[]);
    ~Iqt_application ();

public:
    int get_foo ();
    int foo;
};
#endif
