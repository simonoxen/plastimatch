/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <QProcess>
#include <QTextStream>
#include "pqt_findscu.h"

void
Pqt_findscu::query (QString host, QString port, QString aet)
{
    QTextStream(stdout) 
	<< QString("Performing query: %1 %2 %3\n")
	.arg (host)
	.arg (port)
	.arg (aet);
}

void
Pqt_findscu::debug (void)
{
    QTextStream(stdout) << "Hello world\n";
}
