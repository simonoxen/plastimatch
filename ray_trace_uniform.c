/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif
#include "math_util.h"
#include "ray_trace_exact.h"
#include "timer.h"
#include "volume.h"
#include "volume_limit.h"


void
ray_trace_uniform (
	Volume* vol,			// INPUT: CT Volume
	Volume_limit* vol_limit,	// INPUT: CT volume bounding box
	Ray_trace_callback callback,	// INPUT: Step Action Function
	void* callback_data,		// INPUT: callback data
	double* ip1,			// INPUT: Ray Starting Point
	double* ip2,			// INPUT: Ray Ending Point
	float ray_step			// INPUT: Uniform ray step size
)
{
	// Performed by the function:
	//    1) Take a step along ray
	//    2) Convert location on ray to CT Vol coordinates
	//    3) Invoke callback, passing CT Vol coords
	//    3) Rinse & repeat until ip2 is reached.

	double uv[3];
	double ipx[3];
	double ps[3];

	int ai[3];

	float pix_density;
	double pt;	
	double rlen;
	int idx;
	int z;

	float* img = (float*) vol->img;

	ps[0] = vol->pix_spacing[0];
	ps[1] = vol->pix_spacing[1];
	ps[2] = vol->pix_spacing[2];

	// Get ray length
	rlen = vec3_dist (ip1, ip2);

	// Get unit vector of ray
	vec3_sub3 (uv, ip2, ip1);
	vec3_normalize1 (uv);

	// Trace the ray
	z = 0;
	for (pt = 0; pt < rlen; pt += ray_step)
	{
		// Compute a point along the ray
		ipx[0] = ip1[0] + uv[0] * pt;
		ipx[1] = ip1[1] + uv[1] * pt;
		ipx[2] = ip1[2] + uv[2] * pt;

#if defined (commentout)
		printf ("[ %f %f %f ]\n", ipx[0], ipx[1], ipx[2]);
#endif
		
		// Compute CT Volume coordinates @ point
	        ai[0] = (int) floor ((ipx[0] - vol->offset[0] + 0.5 * ps[0]) / ps[0]);
	        ai[1] = (int) floor ((ipx[1] - vol->offset[1] + 0.5 * ps[1]) / ps[1]);
	        ai[2] = (int) floor ((ipx[2] - vol->offset[2] + 0.5 * ps[2]) / ps[2]);

		idx = ((ai[2]*vol->dim[1] + ai[1]) * vol->dim[0]) + ai[0];
		pix_density = img[idx];
	
		// I am passing the current step along the ray (z) through
		// vox_index here... not exactly great but not horrible.
		(*callback) (callback_data, z++, ray_step, pix_density);
	}
}
