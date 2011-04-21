/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <QApplication>
#include <QLabel>

int
main (int argc, char **argv)
{
    QApplication app(argc, argv);
    QLabel *label = new QLabel("Hello World!");
  
    label->show();

    return app.exec();
}
