/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_io_h_
#define _xio_io_h_

#include "plm_config.h"
#include <string>
#include <vector>
#include "cxt_io.h"

enum Xio_version {
    XIO_VERSION_4_2_1,         /* MGH proton Xio */
    XIO_VERSION_4_33_02,       /* Older MGH photon Xio */
};

enum Xio_patient_position {
    UNKNOWN,
    HFS,
    HFP,
    FFS,
    FFP,
};


plastimatch1_EXPORT 
void
xio_io_get_file_names (
    std::vector<std::pair<std::string,std::string> > *file_names,
    const char *input_dir, 
    const char *regular_expression
);

plastimatch1_EXPORT 
Xio_patient_position
xio_io_patient_position (
    const char *pt_position_str
);
#endif
