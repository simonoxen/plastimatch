/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
   Analyze a vector field for invertibility, smoothness.
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "plmbase.h"
#include "plmsys.h"


void
vf_analyze (Volume* vol)
{
    plm_long i, j, k, v;
    float* img = (float*) vol->img;
    float mean_av[3], mean_v[3];
    float mins[3];
    float maxs[3];

    for (int d = 0; d < 3; d++) {
	mean_av[d] = mean_v[d] = 0.0;
	mins[d] = maxs[d] = img[d];
    }

    for (v = 0, k = 0; k < vol->dim[2]; k++) {
	for (j = 0; j < vol->dim[1]; j++) {
	    for (i = 0; i < vol->dim[0]; i++, v++) {
		float* dxyz = &img[3*v];
		for (int d = 0; d < 3; d++) {
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
    for (int d = 0; d < 3; d++) {
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

/* This is similar to vf_analyze, but works on planar images too */
void
vf_print_stats (Volume* vol)
{
    plm_long i, v;
    int d;
    float mins[3], maxs[3], mean[3];


    mean[0] = mean[1] = mean[2] = (float) 0.0;
    if (vol->pix_type == PT_VF_FLOAT_INTERLEAVED) {
	float *img = (float*) vol->img;
	mins[0] = maxs[0] = img[0];
	mins[1] = maxs[1] = img[1];
	mins[2] = maxs[2] = img[2];
	for (v = 0, i = 0; i < vol->npix; i++) {
	    for (d = 0; d < 3; d++, v++) {
		if (img[v] > maxs[d]) {
		    maxs[d] = img[v];
		} else if (img[v] < mins[d]) {
		    mins[d] = img[v];
		}
		mean[d] += img[v];
	    }
	}
    } else if (vol->pix_type == PT_VF_FLOAT_PLANAR) {
	float **img = (float**) vol->img;
	mins[0] = maxs[0] = img[0][0];
	mins[1] = maxs[1] = img[1][0];
	mins[2] = maxs[2] = img[2][0];
	for (i = 0; i < vol->npix; i++) {
	    for (d = 0; d < 3; d++) {
		if (img[d][i] > maxs[d]) {
		    maxs[d] = img[d][i];
		} else if (img[d][i] < mins[d]) {
		    mins[d] = img[d][i];
		}
		mean[d] += img[d][i];
	    }
	}
    } else {
	printf ("Sorry, vf_print_stats only for vector field volumes\n");
	return;
    }

    for (d = 0; d < 3; d++) {
	mean[d] /= vol->npix;
    }
    printf ("min, mean, max\n");
    for (d = 0; d < 3; d++) {
	printf ("%g %g %g\n", mins[d], mean[d], maxs[d]);
    }
}


void
vf_analyze_strain (Volume* vol)
{
    plm_long i, j, k;
    float* img = (float*) vol->img;
    float total_energy, max_energy;
    float min_dilation = 0.f, max_dilation = 0.f;
    int min_dilation_loc[3] = {0, 0, 0};

    const float LAME_MU = 1.0f;
    const float LAME_NU = 1.0f;

    float di = vol->spacing[0];
    float dj = vol->spacing[1];
    float dk = vol->spacing[2];
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
vf_analyze_jacobian (Volume* vol)
{
    plm_long i, j, k;
    float* img = (float*) vol->img;
    float min_jacobian = 0.f, max_jacobian = 0.f, min_abs_jacobian = 0.f;
    int min_abs_jacobian_loc[3] = {0, 0, 0};;

    float di = vol->spacing[0];
    float dj = vol->spacing[1];
    float dk = vol->spacing[2];
    int first = 1;

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

		// for a zero or constant field u, jacobian must be 1.
		float dui_di = 1 + (0.5 / di) * (dip[0] - din[0]);
		float duj_di = (0.5 / di) * (dip[1] - din[1]);
		float duk_di = (0.5 / di) * (dip[2] - din[2]);
		float dui_dj = (0.5 / dj) * (djp[0] - djn[0]);
		float duj_dj = 1 + (0.5 / dj) * (djp[1] - djn[1]);
		float duk_dj = (0.5 / dj) * (djp[2] - djn[2]);
		float dui_dk = (0.5 / dk) * (dkp[0] - dkn[0]);
		float duj_dk = (0.5 / dk) * (dkp[1] - dkn[1]);
		float duk_dk = 1 + (0.5 / dk) * (dkp[2] - dkn[2]);
		
		float jacobian = 
			+dui_di * ( duj_dj * duk_dk - duj_dk * duk_dj  ) 
			-dui_dj * ( duj_di * duk_dk - duj_dk * duk_di  )
			+dui_dk * ( duj_di * duk_dj - duj_dj * duk_di  )	;
			// absolute value not taken to detect possible sign changes

		if (first) {
			max_jacobian = jacobian;
			min_jacobian = jacobian;
			min_abs_jacobian = fabs(jacobian);
		    min_abs_jacobian_loc[0] = i;
		    min_abs_jacobian_loc[1] = j;
		    min_abs_jacobian_loc[2] = k;
		    first = 0;
		} else {
		    if (fabs(jacobian) < min_abs_jacobian) {
			min_abs_jacobian = fabs(jacobian);
			min_abs_jacobian_loc[0] = i;
			min_abs_jacobian_loc[1] = j;
			min_abs_jacobian_loc[2] = k;
		    };
		    if (jacobian > max_jacobian) max_jacobian = jacobian;
		  	if (jacobian < min_jacobian) min_jacobian = jacobian;
		}

	    }
	}
    }

    printf ("Jacobian: MINJAC %g MAXJAC %g MINABSJAC %g\n", 
	min_jacobian, max_jacobian, min_abs_jacobian);
    printf ("Min abs jacobian at: (%d %d %d)\n", 
	min_abs_jacobian_loc[0], min_abs_jacobian_loc[1], min_abs_jacobian_loc[2]);
}

void
vf_analyze_second_deriv (Volume* vol)
{
    plm_long i, j, k;
    float* img = (float*) vol->img;
    
    float min_sec_der = 0.f, max_sec_der = 0.f, total_sec_der = 0.f;
    int max_sec_der_loc[3] = {0, 0, 0};

    float di = vol->spacing[0];
    float dj = vol->spacing[1];
    float dk = vol->spacing[2];
    int first = 1;

    for (k = 1; k < vol->dim[2]-1; k++) {
	for (j = 1; j < vol->dim[1]-1; j++) {
	    for (i = 1; i < vol->dim[0]-1; i++) {
		
		int v_o = volume_index (vol->dim, i, j, k);

		int vin = volume_index (vol->dim, i-1, j, k);
		int vip = volume_index (vol->dim, i+1, j, k);
		int vjn = volume_index (vol->dim, i, j-1, k);
		int vjp = volume_index (vol->dim, i, j+1, k);
		int vkn = volume_index (vol->dim, i, j, k-1);
		int vkp = volume_index (vol->dim, i, j, k+1);
		
		int vijp = volume_index (vol->dim, i+1, j+1, k);
		int vijn = volume_index (vol->dim, i-1, j-1, k);
		int vikp = volume_index (vol->dim, i+1, j, k+1);
		int vikn = volume_index (vol->dim, i-1, j, k-1);
		int vjkp = volume_index (vol->dim, i, j+1, k+1);
		int vjkn = volume_index (vol->dim, i, j-1, k-1);

		float* d_o = &img[3*v_o];

		float* din = &img[3*vin];
		float* dip = &img[3*vip];
		float* djn = &img[3*vjn];
		float* djp = &img[3*vjp];
		float* dkn = &img[3*vkn];
		float* dkp = &img[3*vkp];

		float *dijp = &img[3*vijp];
		float *dijn = &img[3*vijn];
		float *dikp = &img[3*vikp];
		float *dikn = &img[3*vikn];
		float *djkp = &img[3*vjkp];
		float *djkn = &img[3*vjkn];

		float d2ui_didi = (1./ di) * ( dip[0] - 2 * d_o[0] + din[0] );
		float d2ui_djdj = (1./ dj) * ( djp[0] - 2 * d_o[0] + djn[0] );
		float d2ui_dkdk = (1./ dk) * ( dkp[0] - 2 * d_o[0] + dkn[0] );
		float d2ui_didj = (0.5 / (di*dj))*
		    ( ( dijp[0] + dijn[0] + 2. * d_o[0] ) - 
			( dip[0] + din[0] + djp[0] + djn[0]) );
		float d2ui_didk = (0.5 / (di*dk))*
		    ( ( dikp[0] + dikn[0] + 2. * d_o[0] ) - 
			( dip[0] + din[0] + dkp[0] + dkn[0]) );
		float d2ui_djdk = (0.5 / (dj*dk))*
		    ( ( djkp[0] + djkn[0] + 2. * d_o[0] ) - 
			( djp[0] + djn[0] + dkp[0] + dkn[0]) );

		float d2uj_didi = (1./ di) * ( dip[1] - 2 * d_o[1] + din[1] );
		float d2uj_djdj = (1./ dj) * ( djp[1] - 2 * d_o[1] + djn[1] );
		float d2uj_dkdk = (1./ dk) * ( dkp[1] - 2 * d_o[1] + dkn[1] );
		float d2uj_didj = (0.5 / (di*dj))*
		    ( ( dijp[1] + dijn[1] + 2. * d_o[1] ) - 
			( dip[1] + din[1] + djp[1] + djn[1]) );
		float d2uj_didk = (0.5 / (di*dk))*
		    ( ( dikp[1] + dikn[1] + 2. * d_o[1] ) - 
			( dip[1] + din[1] + dkp[1] + dkn[1]) );
		float d2uj_djdk = (0.5 / (dj*dk))*
		    ( ( djkp[1] + djkn[1] + 2. * d_o[1] ) - 
			( djp[1] + djn[1] + dkp[1] + dkn[1]) );

		float d2uk_didi = (1./ di) * ( dip[2] - 2 * d_o[2] + din[2] );
		float d2uk_djdj = (1./ dj) * ( djp[2] - 2 * d_o[2] + djn[2] );
		float d2uk_dkdk = (1./ dk) * ( dkp[2] - 2 * d_o[2] + dkn[2] );
		float d2uk_didj = (0.5 / (di*dj))*
		    ( ( dijp[2] + dijn[2] + 2. * d_o[2] ) - 
			( dip[2] + din[2] + djp[2] + djn[2]) );
		float d2uk_didk = (0.5 / (di*dk))*
		    ( ( dikp[2] + dikn[2] + 2. * d_o[2] ) - 
			( dip[2] + din[2] + dkp[2] + dkn[2]) );
		float d2uk_djdk = (0.5 / (dj*dk))*
		    ( ( djkp[2] + djkn[2] + 2. * d_o[2] ) - 
			( djp[2] + djn[2] + dkp[2] + dkn[2]) );

		float second_deriv_sq =
		    d2ui_didi*d2ui_didi + d2ui_djdj*d2ui_djdj + d2ui_dkdk*d2ui_dkdk +
		    2*(d2ui_didj*d2ui_didj + d2ui_didk*d2ui_didk + d2ui_djdk*d2ui_djdk) +
		    d2uj_didi*d2uj_didi + d2uj_djdj*d2uj_djdj + d2uj_dkdk*d2uj_dkdk +
		    2*(d2uj_didj*d2uj_didj + d2uj_didk*d2uj_didk + d2uj_djdk*d2uj_djdk) +
		    d2uk_didi*d2uk_didi + d2uk_djdj*d2uk_djdj + d2uk_dkdk*d2uk_dkdk +
		    2*(d2uk_didj*d2uk_didj + d2uk_didk*d2uk_didk + d2uk_djdk*d2uk_djdk) ;

		total_sec_der += second_deriv_sq;

		if (first) {
		    max_sec_der = second_deriv_sq;
		    min_sec_der = second_deriv_sq;
		    max_sec_der_loc[0] = i;
		    max_sec_der_loc[1] = j;
		    max_sec_der_loc[2] = k;
		    first = 0;
		} else {
		    if (second_deriv_sq > max_sec_der) {
			max_sec_der = second_deriv_sq;
			max_sec_der_loc[0] = i;
			max_sec_der_loc[1] = j;
			max_sec_der_loc[2] = k;
		    };
		    if (second_deriv_sq < min_sec_der) min_sec_der = second_deriv_sq;
		}

	    }
	}
    }

    printf ("Second derivatives: MINSECDER %g MAXSECDER %g TOTSECDER %g INTSECDER %g\n", 
	min_sec_der, max_sec_der, total_sec_der,
	total_sec_der * (vol->spacing[0]*vol->spacing[1]*vol->spacing[2]) );
    printf ("Max second derivative: (%d %d %d)\n", 
	max_sec_der_loc[0], max_sec_der_loc[1], max_sec_der_loc[2]);
}

void
vf_analyze_mask (Volume* vol, Volume *mask)
{
    plm_long d, i, j, k, v;
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
    plm_long i, j, k;
    float* img = (float*) vol->img;
    unsigned char*  maskimg = (unsigned char*) mask->img;
    float min_dilation = 0.f, max_dilation = 0.f;
    float total_energy, max_energy;

    const float LAME_MU = 1.0f;
    const float LAME_NU = 1.0f;

    float di = vol->spacing[0];
    float dj = vol->spacing[1];
    float dk = vol->spacing[2];
    int first = 1;

    total_energy = 0.0f;
    max_energy = 0.0f;

    printf ("di = %f dj = %f dk = %f \n", di, dj, dk);
    printf ("vol->dim %ld %ld %ld\n", 
        (long) vol->dim[0], (long) vol->dim[1], (long) vol->dim[2]);
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
