/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_dir_h
#define _xio_dir_h

#include "plm_config.h"
#include "plm_path.h"

typedef struct xio_dir Xio_dir;
struct xio_dir {
    char path[_MAX_PATH];
    
};

plastimatch1_EXPORT
Xio_dir*
xio_dir_create (char *input_dir);

plastimatch1_EXPORT
int
xio_dir_num_patients (Xio_dir* xd);

plastimatch1_EXPORT
void
xio_dir_destroy (Xio_dir* xd);

#endif
