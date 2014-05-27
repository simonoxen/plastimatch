/******************************************************************************
 * odbf.h
 ******************************************************************************
 * dbf Reader and Converter for dBASE files
 * Author: Bjoern Berg <clergyman@gmx.de>
 *
 ******************************************************************************
 * This includes enable dbf to write dBASE files
 ******************************************************************************
 * $Id: odbf.h,v 1.1 2004/09/07 16:04:36 steinm Exp $
 ******************************************************************************/

#ifndef _ODBASE_H_
#define _ODBASE_H_

#include "dbf.h"

#if defined __cplusplus
extern "C" {
#endif

int writeDBFHeader (FILE *fp, P_DBF * p_dbf,
    const char *input, const char *output);
int writeDBFFooter (FILE *fp, P_DBF *p_dbf,
    const char *filename, const char *export_filename);
int writeDBFLine (FILE *fp, P_DBF * p_dbf, const unsigned char *value,
    int header_length, const char *input, const char *output);

#if defined __cplusplus
}
#endif

#endif
