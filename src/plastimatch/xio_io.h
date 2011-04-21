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
    XIO_VERSION_UNKNOWN,
    XIO_VERSION_4_2_1,         /* MGH proton Xio */
    XIO_VERSION_4_33_02,       /* Older MGH photon Xio */
    XIO_VERSION_4_5_0,         /* Current MGH photon Xio */
};


plastimatch1_EXPORT 
void
xio_io_get_file_names (
    std::vector<std::pair<std::string,std::string> > *file_names,
    const char *input_dir, 
    const char *regular_expression
);

#endif
