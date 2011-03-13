/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
    int mi, mj, mk, mv;
    float fxyz[3];
    float* vf_img = (float*) vf->img;
    float* vin_img = (float*) vin->img;
    float* vout_img;

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
		float mxyz[3];

		mxyz[2] = PROJECT_Z(mo_xyz,vin->proj);
		mxyz[1] = PROJECT_Y(mo_xyz,vin->proj);
		mxyz[0] = PROJECT_X(mo_xyz,vin->proj);
		mk = ROUND_INT(mxyz[2]);
		mj = ROUND_INT(mxyz[1]);
		mi = ROUND_INT(mxyz[0]);
		mv = (mk * vin->dim[1] + mj) * vin->dim[0] + mi;

		if (ijk[0] == 0 && ijk[1] == 0 && ijk[2] == 0) {
		    printf ("proj[0] = [%f,%f,%f]\n", 
			vin->proj[0][0],
			vin->proj[0][1],
			vin->proj[0][2]);
		}
		if (ijk[2] == 0 && ijk[1] < 2 && ijk[0] < 2) {
		    printf ("[%f,%f,%f] -> [%f,%f,%f] -> [%f,%f,%f]\n",
			fxyz[0] + dxyz[0], 
			fxyz[1] + dxyz[1], 
			fxyz[2] + dxyz[2], 
			mo_xyz[0], mo_xyz[1], mo_xyz[2],
			mxyz[0], mxyz[1], mxyz[2]);
		}
		if (ijk[0] == 152 && ijk[1] == 180 && ijk[2] == 11) {
		    printf ("[%f,%f,%f] -> [%f,%f,%f] \n"
			"    -> [%f,%f,%f] -> [%f,%f,%f]\n"
			"    -> [%d,%d,%d]\n",
			fxyz[0], fxyz[1], fxyz[2],
			fxyz[0] + dxyz[0], 
			fxyz[1] + dxyz[1], 
			fxyz[2] + dxyz[2], 
			mo_xyz[0], mo_xyz[1], mo_xyz[2],
			mxyz[0], mxyz[1], mxyz[2],
			mi, mj, mk
		    );
		}

		if (mk < 0 || mk >= vin->dim[2]) continue;
		if (mj < 0 || mj >= vin->dim[1]) continue;
		if (mi < 0 || mi >= vin->dim[0]) continue;

		/* Get tri-linear interpolation fractions */
		//li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);

		vout_img[v] = vin_img[mv];
	    }
	}
    }
    return vout;
}
