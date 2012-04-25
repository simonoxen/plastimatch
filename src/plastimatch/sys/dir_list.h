/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dir_list_h_
#define _dir_list_h_

#include "plm_config.h"

typedef struct dir_list Dir_list;
struct dir_list
{
    int num_entries;
    char** entries;
};

#endif
