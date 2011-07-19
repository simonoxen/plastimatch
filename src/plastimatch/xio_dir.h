/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_dir_h
#define _xio_dir_h

#include "plm_config.h"
#include "plm_path.h"

typedef struct xio_studyset_dir Xio_studyset_dir;
struct xio_studyset_dir {
    char path[_MAX_PATH];
};

typedef struct xio_plan_dir Xio_plan_dir;
struct xio_plan_dir {
    char path[_MAX_PATH];
};

typedef struct xio_patient_dir Xio_patient_dir;
struct xio_patient_dir {
    char path[_MAX_PATH];
    int num_studyset_dir;
    int num_plan_dir;
    Xio_studyset_dir *studyset_dir;
    Xio_plan_dir *plan_dir;
};

class plastimatch1_EXPORT Xio_dir {
public:
    Pstring path;
    Pstring m_demographic_fn;
    int num_patient_dir;
    Xio_patient_dir *patient_dir;
public:
    Xio_dir (const char *input_dir);
    ~Xio_dir ();
};

plastimatch1_EXPORT
void
xio_dir_analyze (Xio_dir *xd);

plastimatch1_EXPORT
int
xio_dir_num_patients (Xio_dir* xd);

plastimatch1_EXPORT
Xio_studyset_dir*
xio_plan_dir_get_studyset_dir (Xio_plan_dir* xtpd);

#endif
