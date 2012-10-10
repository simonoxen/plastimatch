/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "plm_math.h"
#include "ray_trace.h"
#include "volume.h"
#include "volume_limit.h"

void
ray_trace_uniform (
    Volume* vol,                  // INPUT: CT Volume
    Volume_limit* vol_limit,      // INPUT: CT volume bounding box
    Ray_trace_callback callback,  // INPUT: Step Action Function
    void* callback_data,          // INPUT: callback data
    double* ip1in,                // INPUT: Ray Starting Point
    double* ip2in,                // INPUT: Ray Ending Point
    float ray_step                // INPUT: Uniform ray step size
)
{
    double uv[3];
    double ipx[3];
    double ps[3];
    double ip1[3];
    double ip2[3];
    double phy_step[3];
    double phy_step_mag;

    int ai[3];

    float pix_density;
    double pt;  
    double rlen;
    int idx;
    size_t z;

    float* img = (float*) vol->img;

    /* Test if ray intersects volume */
    if (!volume_limit_clip_segment (vol_limit, ip1, ip2, ip1in, ip2in)) {
	return;
    }

    ps[0] = vol->spacing[0];
    ps[1] = vol->spacing[1];
    ps[2] = vol->spacing[2];

    // Get ray length
    rlen = vec3_dist (ip1, ip2);

    // Get unit vector of ray
    vec3_sub3 (uv, ip2, ip1);
    vec3_normalize1 (uv);

    phy_step[0] = uv[0] * ray_step;
    phy_step[1] = uv[1] * ray_step;
    phy_step[2] = uv[2] * ray_step;
    phy_step_mag = vec3_len (phy_step);

    // Trace the ray
    z = 0;
    for (pt = 0; pt < rlen; pt += phy_step_mag)
    {
        // Compute a point along the ray
        ipx[0] = ip1[0] + phy_step[0] * z;
        ipx[1] = ip1[1] + phy_step[1] * z;
        ipx[2] = ip1[2] + phy_step[2] * z;

        // Compute CT Volume indices @ point
        ai[0] = (int) floorf ((ipx[0] - vol->offset[0] + 0.5 * ps[0]) / ps[0]);
        ai[1] = (int) floorf ((ipx[1] - vol->offset[1] + 0.5 * ps[1]) / ps[1]);
        ai[2] = (int) floorf ((ipx[2] - vol->offset[2] + 0.5 * ps[2]) / ps[2]);

        idx = ((ai[2]*vol->dim[1] + ai[1]) * vol->dim[0]) + ai[0];
        pix_density = img[idx];

        // I am passing the current step along the ray (z) through
        // vox_index here... not exactly great but not horrible.
//        (*callback) (callback_data, z++, ray_step, pix_density);
        (*callback) (callback_data, z++, phy_step_mag, pix_density);
    }
}
