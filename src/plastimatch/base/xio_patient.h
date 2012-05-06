/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_patient_h
#define _xio_patient_h

#include "plmbase_config.h"
#include "pstring.h"
#include "plm_path.h"

struct Xio_studyset_dir {
    char path[_MAX_PATH];
};

struct Xio_plan_dir {
    char path[_MAX_PATH];
};

/* This class represents a toplevel patient directory */
class API Xio_patient {
public:
    Xio_patient (const char* path);
    ~Xio_patient ();
public:
    Pstring m_path;
    Pstring m_demographic_fn;
    int num_studyset_dir;
    int num_plan_dir;
    Xio_studyset_dir *studyset_dir;
    Xio_plan_dir *plan_dir;
public:
    void add_studyset_dir (std::string studyset_path);
    void add_plan_dir (std::string plan_path);
    void analyze ();
};

#endif
