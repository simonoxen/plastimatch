/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "plm_int.h"
#include "volume.h"

#define round_int(x) ((x)>=0?(long)((x)+0.5):(long)(-(-(x)+0.5)))

gpuit_EXPORT
Volume*
vf_warp (Volume *vout, Volume *vin, Volume *vf)
{
    int d, i, j, k, v;
    int mi, mj, mk, mv;
    float fx, fy, fz;
    float mx, my, mz;
    float* vf_img = (float*) vf->img;
    float* vin_img = (float*) vin->img;
    float* vout_img;

    if (!vout) {
	vout = volume_clone_empty (vin);
    }
    vout_img = (float*) vout->img;
    
    /* Assumes size, spacing of vout same as size, spacing of vf */
    for (d = 0; d < 3; d++) {
	if (vout->dim[d] != vf->dim[d]) {
	    printf("Dimension mismatch between fixed and moving\n");
	    return 0;
	}
	if (vout->pix_spacing[d] != vf->pix_spacing[d]) {
	    printf("Resolutions mismatch between fixed and moving\n");
	    return 0;
	}
	if (vout->offset[d] != vf->offset[d]) {
	    printf("offset mismatch between fixed and moving\n");
	    return 0;
	}
    }

    for (v = 0, k = 0, fz = vf->offset[2]; k < vf->dim[2]; k++, fz += vf->pix_spacing[2]) {
	for (j = 0, fy = vf->offset[1]; j < vf->dim[1]; j++, fy += vf->pix_spacing[1]) {
	    for (i = 0, fx = vf->offset[0]; i < vf->dim[0]; i++, fx += vf->pix_spacing[0], v++) {
		float *dxyz = &vf_img[3*v];
		mz = fz + dxyz[2];
		mk = round_int((mz - vin->offset[2]) / vin->pix_spacing[2]);
		//if (mk < 0 || mk >= vin->dim[2]) continue;
		my = fy + dxyz[1];
		mj = round_int((my - vin->offset[1]) / vin->pix_spacing[1]);
		//if (mj < 0 || mj >= vin->dim[1]) continue;
		mx = fx + dxyz[0];
		mi = round_int((mx - vin->offset[0]) / vin->pix_spacing[0]);
		//if (mi < 0 || mi >= vin->dim[0]) continue;
		mv = (mk * vin->dim[1] + mj) * vin->dim[0] + mi;

		if (i == 128 && j == 128 && (k == 96 || k == 95)) {
		    printf ("(%d %d %d) (%g %g %g) + (%g %g %g) = (%g %g %g) (%d %d %d)\n",
			    i, j, k, fx, fy, fz, dxyz[0], dxyz[1], dxyz[2], 
			    mx, my, mz, mi, mj, mk);
		}
#if defined (commentout)
#endif
		if (mk < 0 || mk >= vin->dim[2]) continue;
		if (mj < 0 || mj >= vin->dim[1]) continue;
		if (mi < 0 || mi >= vin->dim[0]) continue;

		vout_img[v] = vin_img[mv];
	    }
	}
    }
    return vout;
}
