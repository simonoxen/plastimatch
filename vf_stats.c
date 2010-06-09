/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
   Analyze a vector field for invertibility, smoothness.
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "readmha.h"
#include "vf_stats.h"

void
vf_analyze (Volume* vol)
{
    int d, i, j, k, v;
    float* img = (float*) vol->img;
    float mean_av[3], mean_v[3];
    float mins[3];
    float maxs[3];

    for (d = 0; d < 3; d++) {
	mean_av[d] = mean_v[d] = 0.0;
	mins[d] = maxs[d] = img[d];
    }

    for (v = 0, k = 0; k < vol->dim[2]; k++) {
	for (j = 0; j < vol->dim[1]; j++) {
	    for (i = 0; i < vol->dim[0]; i++, v++) {
		float* dxyz = &img[3*v];
		for (d = 0; d < 3; d++) {
		    mean_v[d] += dxyz[d];
		    mean_av[d] += fabs(dxyz[d]);
		    if (dxyz[d] > maxs[d]) {
			maxs[d] = dxyz[d];
		    } else if (dxyz[d] < mins[d]) {
			mins[d] = dxyz[d];
		    }
		}
	    }
	}
    }
    for (d = 0; d < 3; d++) {
	mean_v[d] /= vol->npix;
	mean_av[d] /= vol->npix;
    }

    printf ("Min:       %10.3f %10.3f %10.3f\n", 
	mins[0], mins[1], mins[2]);
    printf ("Mean:      %10.3f %10.3f %10.3f\n", 
	mean_v[0], mean_v[1], mean_v[2]);
    printf ("Max:       %10.3f %10.3f %10.3f\n", 
	maxs[0], maxs[1], maxs[2]);
    printf ("Mean abs:  %10.3f %10.3f %10.3f\n", 
	mean_av[0], mean_av[1], mean_av[2]);
}

void
vf_analyze_strain (Volume* vol)
{
    int i, j, k;
    float* img = (float*) vol->img;
    float total_energy, max_energy;
    float min_dilation, max_dilation;
    int min_dilation_loc[3];

    const float LAME_MU = 1.0f;
    const float LAME_NU = 1.0f;

    float di = vol->pix_spacing[0];
    float dj = vol->pix_spacing[1];
    float dk = vol->pix_spacing[2];
    int first = 1;

    total_energy = 0.0f;
    max_energy = 0.0f;

    for (k = 1; k < vol->dim[2]-1; k++) {
	for (j = 1; j < vol->dim[1]-1; j++) {
	    for (i = 1; i < vol->dim[0]-1; i++) {
		int vin = volume_index (vol->dim, i-1, j, k);
		int vip = volume_index (vol->dim, i+1, j, k);
		int vjn = volume_index (vol->dim, i, j-1, k);
		int vjp = volume_index (vol->dim, i, j+1, k);
		int vkn = volume_index (vol->dim, i, j, k-1);
		int vkp = volume_index (vol->dim, i, j, k+1);

		float* din = &img[3*vin];
		float* dip = &img[3*vip];
		float* djn = &img[3*vjn];
		float* djp = &img[3*vjp];
		float* dkn = &img[3*vkn];
		float* dkp = &img[3*vkp];

		float dui_di = (0.5 / di) * (dip[0] - din[0]);
		float duj_di = (0.5 / di) * (dip[1] - din[1]);
		float duk_di = (0.5 / di) * (dip[2] - din[2]);
		float dui_dj = (0.5 / dj) * (djp[0] - djn[0]);
		float duj_dj = (0.5 / dj) * (djp[1] - djn[1]);
		float duk_dj = (0.5 / dj) * (djp[2] - djn[2]);
		float dui_dk = (0.5 / dk) * (dkp[0] - dkn[0]);
		float duj_dk = (0.5 / dk) * (dkp[1] - dkn[1]);
		float duk_dk = (0.5 / dk) * (dkp[2] - dkn[2]);
		
		float e_ii = dui_di;
		float e_jj = duj_dj;
		float e_kk = duk_dk;
		float e_ij = 0.5 * (dui_dj + duj_di);
		float e_jk = 0.5 * (duj_dk + duk_dj);
		float e_ki = 0.5 * (duk_di + dui_dk);

		float dilation = e_ii + e_jj + e_kk;
		float shear = dilation 
		    + 2.0f * (e_ij * e_ij + e_jk * e_jk + e_ki * e_ki);
		
		float energy = 0.5 * LAME_NU * dilation * dilation
		    + LAME_MU * shear;

		total_energy += energy;
		if (energy > max_energy) max_energy = energy;

		if (first) {
		    min_dilation = dilation;
		    max_dilation = dilation;
		    min_dilation_loc[0] = i;
		    min_dilation_loc[1] = j;
		    min_dilation_loc[2] = k;
		    first = 0;
		} else {
		    if (dilation < min_dilation) {
			min_dilation = dilation;
			min_dilation_loc[0] = i;
			min_dilation_loc[1] = j;
			min_dilation_loc[2] = k;
		    }
		    else if (dilation > max_dilation) {
			max_dilation = dilation;
		    }
		}
	    }
	}
    }

    printf ("Energy: MINDIL %g MAXDIL %g MAXSTRAIN %g TOTSTRAIN %g\n", 
	min_dilation, max_dilation, max_energy, total_energy);
    printf ("Min dilation at: (%d %d %d)\n", 
	min_dilation_loc[0], min_dilation_loc[1], min_dilation_loc[2]);
}

void
vf_analyze_mask (Volume* vol, Volume *mask)
{
    int d, i, j, k, v;
    int mask_npixels = 0;
    float* img = (float*) vol->img;
    unsigned char* maskimg = (unsigned char*) mask->img;
    float mean_av[3], mean_v[3];
    float mins[3];
    float maxs[3];

    for (d = 0; d < 3; d++) {
	mean_av[d] = mean_v[d] = 0.0;
	mins[d] = maxs[d] = img[d];
    }

    for (v = 0, k = 0; k < vol->dim[2]; k++) {
	for (j = 0; j < vol->dim[1]; j++) {
	    for (i = 0; i < vol->dim[0]; i++, v++) {
		float* dxyz = &img[3*v];
                unsigned int maskval = (unsigned int) maskimg[v];
                if (maskval > 0) {
		    mask_npixels ++;
		    for (d = 0; d < 3; d++) {
		        mean_v[d] += dxyz[d];
		        mean_av[d] += fabs(dxyz[d]);
		        if (dxyz[d] > maxs[d]) {
			    maxs[d] = dxyz[d];
		        } else if (dxyz[d] < mins[d]) {
			    mins[d] = dxyz[d];
                        }
		    }
                }
	    }
	}
    }
    for (d = 0; d < 3; d++) {
	mean_v[d] /= mask_npixels;
	mean_av[d] /= mask_npixels;
    }

    printf ("Min within mask:       %10.3f %10.3f %10.3f\n", mins[0], mins[1], mins[2]);
    printf ("Mean within mask:      %10.3f %10.3f %10.3f\n", mean_v[0], mean_v[1], mean_v[2]);
    printf ("Max within mask:       %10.3f %10.3f %10.3f\n", maxs[0], maxs[1], maxs[2]);
    printf ("Mean abs within mask:  %10.3f %10.3f %10.3f\n", mean_av[0], mean_av[1], mean_av[2]);
}

void
vf_analyze_strain_mask (Volume* vol, Volume* mask)
{
    int i, j, k;
    float* img = (float*) vol->img;
    unsigned char*  maskimg = (unsigned char*) mask->img;
    float min_dilation, max_dilation;
    float total_energy, max_energy;

    const float LAME_MU = 1.0f;
    const float LAME_NU = 1.0f;

    float di = vol->pix_spacing[0];
    float dj = vol->pix_spacing[1];
    float dk = vol->pix_spacing[2];
    int first = 1;

    total_energy = 0.0f;
    max_energy = 0.0f;

    printf ("di = %f dj = %f dk = %f \n", di, dj, dk);
    printf ("vol->dim %d %d %d\n", vol->dim[0], vol->dim[1], vol->dim[2]);
    for (k = 1; k < vol->dim[2]-1; k++) {
	for (j = 1; j < vol->dim[1]-1; j++) {
	    for (i = 1; i < vol->dim[0]-1; i++) {
		int vin = volume_index (vol->dim, i-1, j, k);
		int vip = volume_index (vol->dim, i+1, j, k);
		int vjn = volume_index (vol->dim, i, j-1, k);
		int vjp = volume_index (vol->dim, i, j+1, k);
		int vkn = volume_index (vol->dim, i, j, k-1);
		int vkp = volume_index (vol->dim, i, j, k+1);

                unsigned int  maskval_in = (unsigned int) maskimg[vin];
                unsigned int  maskval_ip = (unsigned int) maskimg[vip];
                unsigned int  maskval_jn = (unsigned int) maskimg[vjn];
                unsigned int  maskval_jp = (unsigned int) maskimg[vjp];
                unsigned int  maskval_kn = (unsigned int) maskimg[vkn];
                unsigned int  maskval_kp = (unsigned int) maskimg[vkp];


                if ( ( maskval_in > 0  &&  maskval_ip > 0 )  &&
                     ( maskval_jn > 0  &&  maskval_jp > 0 ) &&
                     ( maskval_kn > 0  &&  maskval_kp > 0 ) ) {

		    float* din = &img[3*vin];
		    float* dip = &img[3*vip];
		    float* djn = &img[3*vjn];
		    float* djp = &img[3*vjp];
		    float* dkn = &img[3*vkn];
		    float* dkp = &img[3*vkp];

		    float dui_di = (0.5 / di) * (dip[0] - din[0]);
		    float duj_di = (0.5 / di) * (dip[1] - din[1]);
		    float duk_di = (0.5 / di) * (dip[2] - din[2]);
		    float dui_dj = (0.5 / dj) * (djp[0] - djn[0]);
		    float duj_dj = (0.5 / dj) * (djp[1] - djn[1]);
		    float duk_dj = (0.5 / dj) * (djp[2] - djn[2]);
		    float dui_dk = (0.5 / dk) * (dkp[0] - dkn[0]);
		    float duj_dk = (0.5 / dk) * (dkp[1] - dkn[1]);
		    float duk_dk = (0.5 / dk) * (dkp[2] - dkn[2]);

		    float e_ii = dui_di;
		    float e_jj = duj_dj;
		    float e_kk = duk_dk;
		    float e_ij = 0.5 * (dui_dj + duj_di);
		    float e_jk = 0.5 * (duj_dk + duk_dj);
		    float e_ki = 0.5 * (duk_di + dui_dk);

		    float dilation = e_ii + e_jj + e_kk;
		    float shear = dilation 
		        + 2.0f * (e_ij * e_ij + e_jk * e_jk + e_ki * e_ki);
		
		    float energy = 0.5 * LAME_NU * dilation * dilation
		        + LAME_MU * shear;

		    total_energy += energy;
		    if (energy > max_energy) max_energy = energy;

		    if (first) {
		        min_dilation = dilation;
		        max_dilation = dilation;
		        first = 0;
		    } else {
		        if (dilation < min_dilation) min_dilation = dilation;
		        else if (dilation > max_dilation) max_dilation = dilation;
		    }
	        }
            }
	}
    }

    printf ("Energy: MINDIL %g MAXDIL %g MAXSTRAIN %g TOTSTRAIN %g\n", 
	min_dilation, max_dilation, max_energy, total_energy);
}
