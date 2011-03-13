/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interpolate.h"
#include "math_util.h"
#include "plm_config.h"
#include "plm_int.h"
#include "vf.h"
#include "volume.h"
#include "volume_macros.h"

Volume*
vf_warp (Volume *vout, Volume *vin, Volume *vf)
{
    int d, v, ijk[3];
    float fxyz[3];
    float* vf_img = (float*) vf->img;
    float* vout_img;
    float* m_img = (float*) vin->img;

    printf ("Direction cosines: "
	"vin = %f %f %f ...\n"
	"vf = %f %f %f ...\n",
	vin->direction_cosines[0],
	vin->direction_cosines[1],
	vin->direction_cosines[2],
	vf->direction_cosines[0],
	vf->direction_cosines[1],
	vf->direction_cosines[2]
    );

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

    for (v = 0, LOOP_Z (ijk, fxyz, vf)) {
	for (LOOP_Y (ijk, fxyz, vf)) {
	    for (LOOP_X (ijk, fxyz, vf), v++) {
		float *dxyz = &vf_img[3*v];
		float mo_xyz[3] = {
		    fxyz[0] + dxyz[0] - vin->offset[0],
		    fxyz[1] + dxyz[1] - vin->offset[1],
		    fxyz[2] + dxyz[2] - vin->offset[2]
		};
		float m_val;
		float li_1[3];  /* Fraction of interpolant in lower index */
		float li_2[3];  /* Fraction of interpolant in upper index */
		float mijk[3];
		int mijk_r[3], mijk_f[3];
		int mvf;

		mijk[2] = PROJECT_Z(mo_xyz,vin->proj);
		if (mijk[2] < -0.5 || mijk[2] > vin->dim[2] - 0.5) continue;
		mijk[1] = PROJECT_Y(mo_xyz,vin->proj);
		if (mijk[1] < -0.5 || mijk[1] > vin->dim[1] - 0.5) continue;
		mijk[0] = PROJECT_X(mo_xyz,vin->proj);
		if (mijk[0] < -0.5 || mijk[0] > vin->dim[0] - 0.5) continue;

#if defined (commentout)
		/* Nearest neighbor */
		mijk_r[2] = ROUND_INT(mijk[2]);
		mijk_r[1] = ROUND_INT(mijk[1]);
		mijk_r[0] = ROUND_INT(mijk[0]);
		mv = (mk * vin->dim[1] + mj) * vin->dim[0] + mi;
		if (mk < 0 || mk >= vin->dim[2]) continue;
		if (mj < 0 || mj >= vin->dim[1]) continue;
		if (mi < 0 || mi >= vin->dim[0]) continue;
		vout_img[v] = vin_img[mv];
#endif

		/* Get tri-linear interpolation fractions */
		li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, vin);

		/* Find linear index of "corner voxel" in moving image */
		mvf = INDEX_OF (mijk_f, vin->dim);
		
		/* Compute moving image intensity using linear interpolation */
		LI_VALUE (m_val, 
		    li_1[0], li_2[0],
		    li_1[1], li_2[1],
		    li_1[2], li_2[2],
		    mvf, m_img, vin);

		/* Assign the value */
		vout_img[v] = m_val;
	    }
	}
    }
    return vout;
}
