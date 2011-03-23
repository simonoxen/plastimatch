/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _simplify_points_h_
#define _simplify_points_h_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <iterator>
#include <set>
#include <algorithm>
#include "itk_pointset.h"
#include "plm_config.h"
#include "plm_file_format.h"
#include "rtds.h"
#include "rtss_polyline_set.h"
#include "vnl/vnl_random.h"

plastimatch1_EXPORT 
void do_simplify(Rtds *rtds, int percentage);
//void do_simplify(Rtds *rtds, Plm_file_format file_type,int percentage);

#endif
