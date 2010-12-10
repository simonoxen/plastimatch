/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _simplify_points_h_
#define _simplify_points_h_


#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <iterator>
#include <set>
#include <algorithm>

//#include "file_util.h"
//#include "gdcm_rtss.h"
//#include "getopt.h"
#include "plm_file_format.h"
//#include "plm_image_header.h"
//#include "plm_image_patient_position.h"
//#include "plm_warp.h"
#include "rtds.h"
//#include "rtds_warp.h"
#include "rtss_polyline_set.h"
#include "ss_image.h"
#include "vnl/vnl_random.h"

//typedef itk::Vector < float, 3 > Points3Type;
typedef itk::PointSet< short, 3 > PointSetSimplifyType;
typedef PointSetSimplifyType::PointsContainer PointsSimplifyContainer;

plastimatch1_EXPORT 
void do_simplify(Rtds *rtds, Plm_file_format file_type,int percentage);

#endif
