/*****************************************************************************
 * statistic.h
 * inherits the statistic functions for dBASE files
 * Author: Bjoern Berg, September 2002
 * Email: clergyman@gmx.de
 * dbf Reader and Converter for dBase III, IV, 5.0 
 *
 * see statistic.c for history and details
 ****************************************************************************/

#ifndef _DBF_STATS_
#define _DBF_STATS_
#include "congraph.h"
#include "libdbf.h"

#if defined __cplusplus
extern "C" {
#endif

void dbf_file_info (P_DBF *p_dbf);
void dbf_field_stat (P_DBF *p_dbf);

#if defined __cplusplus
}
#endif

#endif
