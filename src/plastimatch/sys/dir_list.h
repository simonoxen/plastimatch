/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dir_list_h_
#define _dir_list_h_

#include "plmsys_config.h"
#include <string>

class PLMSYS_API Dir_list
{
public:
    int num_entries;
    char** entries;
public:
    Dir_list ();
    Dir_list (const char* dir);
    Dir_list (const std::string& dir);
    ~Dir_list ();

    void init ();
    void load (const char* dir);
};

/* logfile.cxx */
PLMSYS_C_API void logfile_open (char* log_fn);
PLMSYS_C_API void logfile_close (void);
PLMSYS_C_API void logfile_printf (const char* fmt, ...);
#define lprintf logfile_printf


#endif
