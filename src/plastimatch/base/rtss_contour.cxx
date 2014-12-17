/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"

#include "logfile.h"
#include "plm_math.h"
#include "rtss_contour.h"

Rtss_contour::Rtss_contour ()
{
    this->slice_no = -1;
    this->ct_slice_uid = "";
    this->num_vertices = 0;
    this->x = 0;
    this->y = 0;
    this->z = 0;
}

Rtss_contour::~Rtss_contour ()
{
    free (this->x);
    free (this->y);
    free (this->z);

    this->slice_no = -1;
    this->ct_slice_uid = "";
    this->num_vertices = 0;
    this->x = 0;
    this->y = 0;
    this->z = 0;
}

void
Rtss_contour::find_direction_cosines ()
{
    /* Need at least three points to find slice plane */
    if (this->num_vertices < 3) {
        lprintf ("Failed to find DC, not enough points\n");
        return;
    }

    /* Find triangle with long legs using greedy search */
    float p1[3], p2[3], p3[3];
    p1[0] = this->x[0];
    p1[1] = this->y[0];
    p1[2] = this->z[0];
    p2[0] = this->x[1];
    p2[1] = this->y[1];
    p2[2] = this->z[1];
    p3[0] = this->x[2];
    p3[1] = this->y[2];
    p3[2] = this->z[2];
    float min_dist = std::min (vec3_distsq (p1, p2), 
        std::min (vec3_distsq (p1, p3), vec3_distsq (p2, p3)));
    for (int k = 3; k < this->num_vertices; k++) {
        /* do something */
    }
}
