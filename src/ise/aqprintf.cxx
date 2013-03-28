/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdarg.h>
#include <stdio.h>
#include <QString>
#include <QTime>


/* Just a printf, but also flushes the stream */
int 
aqprintf (const char * format, ...)
{
    //add time tag in format
    //QString* strInputQ = new QString();
    //strInputQ->append(format);
    //std::string strInputStd = strInputQ->toStdString();
    //const char* format2 = (strInputQ->toStdString()).c_str();

    //char* str = new char [512];
    //memset(str, 0, 512);    
    //strcpy(str, format);

    va_list argptr;
    int rc;   

    va_start (argptr, format);
    rc = vprintf (format, argptr);

    va_end (argptr);
    fflush (stdout);
    return rc;
}
//
//void
//PrintCurrentTime()
//{
//    QTime time = QTime::currentTime();    
//    QString str = time.toString("@hh:mm:ss.zzz\n");    
//    aqprintf(str.toLocal8Bit().constData());
//}