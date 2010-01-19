/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_io_h_
#define _xio_io_h_

#include "plm_config.h"
#include <string>
#include <vector>
#include "cxt_io.h"

plastimatch1_EXPORT 
void
xio_io_get_file_names (
    std::vector<std::pair<std::string,std::string> > *file_names,
    const char *input_dir, 
    const char *regular_expression
);

#endif
