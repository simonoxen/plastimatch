/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pqt_database_h_
#define _pqt_database_h_

#include "plm_config.h"
#include <QString>
#include <QSqlQuery>

void
pqt_database_start (QString db_path);
QSqlQuery
pqt_database_query_data_source_label (void);
void
pqt_database_stop (void);

#endif
