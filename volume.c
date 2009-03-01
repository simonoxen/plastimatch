/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "volume.h"

#define round_int(x) ((x)>=0?(long)((x)+0.5):(long)(-(-(x)+0.5)))

int
volume_index (int* dims, int i, int j, int k)
{
    return i + (dims[0] * (j + dims[1] * k));
}

Volume*
volume_create (int* dim, float* offset, float* pix_spacing, 
	       enum Pixel_Type pix_type, int min_size)
{
    int i;
    Volume* vol = (Volume*) malloc (sizeof(Volume));
    if (!vol) {
	fprintf (stderr, "Memory allocation failed.\n");
	exit(1);
    }

    for (i = 0; i < 3; i++) {
	vol->dim[i] = dim[i];
	vol->offset[i] = offset[i];
	vol->pix_spacing[i] = pix_spacing[i];
    }
    vol->npix = vol->dim[0] * vol->dim[1] * vol->dim[2];
    vol->pix_type = pix_type;

    switch (pix_type) {
    case PT_UCHAR:
	vol->pix_size = sizeof(unsigned char);
	break;
    case PT_SHORT:
	vol->pix_size = sizeof(short);
	break;
    case PT_FLOAT:
	vol->pix_size = sizeof(float);
	break;
    case PT_VF_FLOAT_INTERLEAVED:
	vol->pix_size = 3 * sizeof(float);
	break;
    case PT_VF_FLOAT_PLANAR:
	vol->pix_size = sizeof(float);
	break;
    default:
	fprintf (stderr, "Unhandled type in volume_create().\n");
	exit (-1);
    }

    /* Compute some auxiliary variables */
    vol->xmin = vol->offset[0] - vol->pix_spacing[0] / 2;
    vol->xmax = vol->xmin + vol->pix_spacing[0] * vol->dim[0];
    vol->ymin = vol->offset[1] - vol->pix_spacing[1] / 2;
    vol->ymax = vol->ymin + vol->pix_spacing[1] * vol->dim[1];
    vol->zmin = vol->offset[2] - vol->pix_spacing[2] / 2;
    vol->zmax = vol->zmin + vol->pix_spacing[2] * vol->dim[2];

    if (pix_type == PT_VF_FLOAT_PLANAR) {
	int i;
	int alloc_size = min_size;
	float** der = (float**) malloc (3*sizeof(float*));
	if (!der) {
	    fprintf (stderr, "Memory allocation failed.\n");
	    exit(1);
	}
	if (alloc_size < vol->npix) alloc_size = vol->npix;
	printf ("alloc_size = %d, min_size = %d, npix = %d\n", alloc_size, min_size, vol->npix);
	for (i=0; i < 3; i++) {
	    der[i] = (float*) malloc (alloc_size*sizeof(float));
	    if (!der[i]) {
		fprintf (stderr, "Memory allocation failed.\n");
		exit(1);
	    }
	    memset(der[i], 0, alloc_size*sizeof(float));
	}
	vol->img = (void*) der;
    } else {
	vol->img = (void*) malloc (vol->pix_size * vol->npix);
	if (!vol->img) {
	    fprintf (stderr, "Memory allocation failed (alloc size = %d).\n",
		    vol->pix_size * vol->npix);
	    exit(1);
	}
	memset (vol->img, 0, vol->pix_size * vol->npix);
    }

    return vol;
}

Volume*
volume_clone_empty (Volume* ref)
{
    Volume* vout;
    vout = volume_create (ref->dim, ref->offset, ref->pix_spacing, ref->pix_type, 0);
    return vout;
}

Volume*
volume_clone (Volume* ref)
{
    Volume* vout;
    vout = volume_create (ref->dim, ref->offset, ref->pix_spacing, ref->pix_type, 0);
    switch (ref->pix_type) {
    case PT_UCHAR:
    case PT_SHORT:
    case PT_FLOAT:
    case PT_VF_FLOAT_INTERLEAVED:
	memcpy (vout->img, ref->img, ref->npix * ref->pix_size);
	break;
    case PT_VF_FLOAT_PLANAR:
    default:
	fprintf (stderr, "Unsupported clone\n");
	exit (-1);
	break;
    }
    return vout;
}

void
volume_free (Volume* vol)
{
    if (vol->pix_type == PT_VF_FLOAT_PLANAR) {
	float** planes = (float**) vol->img;
	free (planes[0]);
	free (planes[1]);
	free (planes[2]);
    }
    free (vol->img);
    free (vol);
}

void
volume_convert_to_float (Volume* ref)
{
    switch (ref->pix_type) {
    case PT_UCHAR:
	fprintf (stderr, "Sorry, UCHAR to FLOAT is not implemented\n");
	exit (-1);
	break;
    case PT_SHORT:
	{
	    int v;
	    short* old_img;
	    float* new_img;

	    old_img = (short*) ref->img;
	    new_img = (float*) malloc (sizeof(float) * ref->npix);
	    if (!new_img) {
		fprintf (stderr, "Memory allocation failed.\n");
		exit(1);
	    }
	    for (v = 0; v < ref->npix; v++) {
		new_img[v] = (float) old_img[v];
	    }
	    ref->pix_size = sizeof(float);
	    ref->pix_type = PT_FLOAT;
	    ref->img = (void*) new_img;
	    free (old_img);
	}
	break;
    case PT_FLOAT:
	/* Nothing to do */
	break;
    case PT_VF_FLOAT_INTERLEAVED:
    case PT_VF_FLOAT_PLANAR:
    default:
	/* Can't convert this */
	fprintf (stderr, "Sorry, unsupported conversion to FLOAT\n");
	exit (-1);
	break;
    }
}

void
volume_convert_to_short (Volume* ref)
{
    switch (ref->pix_type) {
    case PT_UCHAR:
	fprintf (stderr, "Sorry, UCHAR to SHORT is not implemented\n");
	exit (-1);
	break;
    case PT_SHORT:
	/* Nothing to do */
	break;
    case PT_FLOAT:
	{
	    int v;
	    float* old_img;
	    short* new_img;

	    old_img = (float*) ref->img;
	    new_img = (short*) malloc (sizeof(short) * ref->npix);
	    if (!new_img) {
		fprintf (stderr, "Memory allocation failed.\n");
		exit(1);
	    }
	    for (v = 0; v < ref->npix; v++) {
		new_img[v] = (short) old_img[v];
	    }
	    ref->pix_size = sizeof(short);
	    ref->pix_type = PT_SHORT;
	    ref->img = (void*) new_img;
	    free (old_img);
	}
	break;
    case PT_VF_FLOAT_INTERLEAVED:
    case PT_VF_FLOAT_PLANAR:
    default:
	/* Can't convert this */
	fprintf (stderr, "Sorry, unsupported conversion to SHORT\n");
	exit (-1);
	break;
    }
}

void
vf_convert_to_interleaved (Volume* vf)
{
    switch (vf->pix_type) {
    case PT_VF_FLOAT_INTERLEAVED:
	/* Nothing to do */
	break;
    case PT_VF_FLOAT_PLANAR:
	{
	    int v;
	    float** planar = (float**) vf->img;
	    float* inter = (float*) malloc (3*sizeof(float*)*vf->npix);
	    if (!inter) {
		fprintf (stderr, "Memory allocation failed.\n");
		exit(1);
	    }
	    for (v = 0; v < vf->npix; v++) {
		inter[3*v + 0] = planar[0][v];
		inter[3*v + 1] = planar[1][v];
		inter[3*v + 2] = planar[2][v];
	    }
	    free (planar[0]);
	    free (planar[1]);
	    free (planar[2]);
	    free (planar);
	    vf->img = (void*) inter;
	    vf->pix_type = PT_VF_FLOAT_INTERLEAVED;
	    vf->pix_size = 3*sizeof(float);
	}
	break;
    case PT_UCHAR:
    case PT_SHORT:
    case PT_FLOAT:
    default:
	/* Can't convert this */
	fprintf (stderr, "Sorry, unsupported conversion to VF\n");
	exit (-1);
	break;
    }
}

/* min_size allows us to pad the size of the plane for brook stream reads */
void
vf_convert_to_planar (Volume* ref, int min_size)
{
    switch (ref->pix_type) {
    case PT_VF_FLOAT_INTERLEAVED:
	{
	    int i;
	    float* img = (float*) ref->img;
	    int alloc_size = min_size;
	    float** der = (float**) malloc (3*sizeof(float*));
	    if (!der) {
		printf ("Memory allocation failed.\n");
		exit(1);
	    }
	    if (alloc_size < ref->npix) alloc_size = ref->npix;
	    printf ("alloc_size = %d, min_size = %d, npix = %d\n", alloc_size, min_size, ref->npix);
	    for (i=0; i < 3; i++) {
		der[i] = (float*) malloc (alloc_size*sizeof(float));
		if (!der[i]) {
		    printf ("Memory allocation failed for stage 2, dimension %d for current velocity..........Exiting\n",i);
		    exit(1);
		}
	    }
	    for (i = 0; i < ref->npix; i++) {
		der[0][i] = img[3*i + 0];
		der[1][i] = img[3*i + 1];
		der[2][i] = img[3*i + 2];
	    }
	    free (ref->img);
	    ref->img = (void*) der;
	    ref->pix_type = PT_VF_FLOAT_PLANAR;
	    ref->pix_size = sizeof(float);
	}
        break;
    case PT_VF_FLOAT_PLANAR:
	/* Nothing to do */
	break;
    case PT_UCHAR:
    case PT_SHORT:
    case PT_FLOAT:
    default:
	/* Can't convert this */
	fprintf (stderr, "Sorry, unsupportd conversion to VF\n");
	exit (-1);
	break;
    }
}

/*
void
vf_pad_planar (Volume* vol, int size)
{
    int i;
    float** img = (float**) vol->img;

    if (vol->pix_type != PT_VF_FLOAT_PLANAR) {
	fprintf (stderr, "Error, can't pad_planar a non-planar volume\n");
	exit (-1);
    }

    for (i = 0; i < 3; i++) {
	img[i] = realloc (img[i], size);
    }
}
*/

/* Nearest neighbor interpolation */
static Volume*
volume_resample_float (Volume* vol_in, int* dim, 
		      float* offset, float* pix_spacing)
{
    int i, j, k, v;
    float x, y, z;
    float x_in, y_in, z_in;
    int xidx, yidx, zidx;
    Volume* vol_out;
    float *in_img, *out_img;
    float val;
    float default_val = 0.0f;

    vol_out = volume_create (dim, offset, pix_spacing, PT_FLOAT, 0);
    in_img = (float*) vol_in->img;
    out_img = (float*) vol_out->img;

    for (k = 0, v = 0, z = offset[2]; k < dim[2]; k++, z += pix_spacing[2]) {
	z_in = (z - vol_in->offset[2]) / vol_in->pix_spacing[2];
	zidx = round_int (z_in);
	for (j = 0, y = offset[1]; j < dim[1]; j++, y += pix_spacing[1]) {
	    y_in = (y - vol_in->offset[1]) / vol_in->pix_spacing[1];
	    yidx = round_int (y_in);
	    for (i = 0, x = offset[0]; i < dim[0]; i++, x += pix_spacing[0], v++) {
		x_in = (x - vol_in->offset[0]) / vol_in->pix_spacing[0];
		xidx = round_int (x_in);
		if (zidx < 0 || zidx >= vol_in->dim[2] || yidx < 0 || yidx >= vol_in->dim[1] || xidx < 0 || xidx >= vol_in->dim[0]) {
		    val = default_val;
		} else {
		    int idx = zidx*vol_in->dim[1]*vol_in->dim[0] + yidx*vol_in->dim[0] + xidx;
		    val = in_img[idx];
		}
		out_img[v] = val;
	    }
	}
    }

    return vol_out;
}

/* Nearest neighbor interpolation */
static Volume*
volume_resample_vf_float_interleaved (Volume* vol_in, int* dim, 
				      float* offset, float* pix_spacing)
{
    int d, i, j, k, v;
    float x, y, z;
    float x_in, y_in, z_in;
    int xidx, yidx, zidx;
    Volume* vol_out;
    float *in_img, *out_img;
    float* val;
    float default_val[3] = { 0.0f, 0.0f, 0.0f };

    vol_out = volume_create (dim, offset, pix_spacing, PT_VF_FLOAT_INTERLEAVED, 0);
    in_img = (float*) vol_in->img;
    out_img = (float*) vol_out->img;

    for (k = 0, v = 0, z = offset[2]; k < dim[2]; k++, z += pix_spacing[2]) {
	z_in = (z - vol_in->offset[2]) / vol_in->pix_spacing[2];
	zidx = round_int (z_in);
	for (j = 0, y = offset[1]; j < dim[1]; j++, y += pix_spacing[1]) {
	    y_in = (y - vol_in->offset[1]) / vol_in->pix_spacing[1];
	    yidx = round_int (y_in);
	    for (i = 0, x = offset[0]; i < dim[0]; i++, x += pix_spacing[0]) {
		x_in = (x - vol_in->offset[0]) / vol_in->pix_spacing[0];
		xidx = round_int (x_in);
		if (zidx < 0 || zidx >= vol_in->dim[2] || yidx < 0 || yidx >= vol_in->dim[1] || xidx < 0 || xidx >= vol_in->dim[0]) {
		    val = default_val;
		} else {
		    int idx = zidx*vol_in->dim[1]*vol_in->dim[0] + yidx*vol_in->dim[0] + xidx;
		    val = &in_img[idx*3];
		}
		for (d = 0; d < 3; d++, v++) {
		    out_img[v] = val[d];
		}
	    }
	}
    }

    return vol_out;
}

/* Nearest neighbor interpolation */
static Volume*
volume_resample_vf_float_planar (Volume* vol_in, int* dim, 
			      float* offset, float* pix_spacing)
{
    int d, i, j, k, v;
    float x, y, z;
    float x_in, y_in, z_in;
    int xidx, yidx, zidx;
    Volume* vol_out;
    float **in_img, **out_img;

    vol_out = volume_create (dim, offset, pix_spacing, PT_VF_FLOAT_PLANAR, 0);
    in_img = (float**) vol_in->img;
    out_img = (float**) vol_out->img;

    for (k = 0, v = 0, z = offset[2]; k < dim[2]; k++, z += pix_spacing[2]) {
	z_in = (z - vol_in->offset[2]) / vol_in->pix_spacing[2];
	zidx = round_int (z_in);
	for (j = 0, y = offset[1]; j < dim[1]; j++, y += pix_spacing[1]) {
	    y_in = (y - vol_in->offset[1]) / vol_in->pix_spacing[1];
	    yidx = round_int (y_in);
	    for (i = 0, x = offset[0]; i < dim[0]; i++, x += pix_spacing[0], v++) {
		x_in = (x - vol_in->offset[0]) / vol_in->pix_spacing[0];
		xidx = round_int (x_in);
		if (zidx < 0 || zidx >= vol_in->dim[2] || yidx < 0 || yidx >= vol_in->dim[1] || xidx < 0 || xidx >= vol_in->dim[0]) {
		    for (d = 0; d < 3; d++) {
			out_img[d][v] = 0.0;		/* Default value */
		    }
		} else {
		    for (d = 0; d < 3; d++) {
			int idx = zidx*vol_in->dim[1]*vol_in->dim[0] + yidx*vol_in->dim[0] + xidx;
			out_img[d][v] = in_img[d][idx];
		    }
		}
	    }
	}
    }

    return vol_out;
}

Volume*
volume_resample (Volume* vol_in, int* dim, float* offset, float* pix_spacing)
{
    switch (vol_in->pix_type) {
	case PT_UCHAR:
	case PT_SHORT:
	    fprintf (stderr, "Error, resampling PT_SHORT or PT_UCHAR is unsupported\n");
	    return 0;
	case PT_FLOAT:
	    return volume_resample_float (vol_in, dim, offset, pix_spacing);
	case PT_VF_FLOAT_INTERLEAVED:
	    return volume_resample_vf_float_interleaved (vol_in, dim, offset, pix_spacing);
	case PT_VF_FLOAT_PLANAR:
	    return volume_resample_vf_float_planar (vol_in, dim, offset, pix_spacing);
	default:
	    fprintf (stderr, "Error, unknown pix_type: %d\n", vol_in->pix_type);
	    return 0;
    }
}

Volume*
volume_subsample (Volume* vol_in, int* sampling_rate)
{
    int d;
    int dim[3];
    float offset[3];
    float pix_spacing[3];

    for (d = 0; d < 3; d++) {
	float in_size = vol_in->dim[d] * vol_in->pix_spacing[d];

	dim[d] = vol_in->dim[d] / sampling_rate[d];
	if (dim[d] < 1) dim[d] = 1;
	pix_spacing[d] = in_size / dim[d];
	offset[d] = (float) (vol_in->offset[d] - 0.5 * vol_in->pix_spacing[d] + 0.5 * pix_spacing[d]);
    }
    return volume_resample (vol_in, dim, offset, pix_spacing);
}

void
vf_print_stats (Volume* vol)
{
    int i, d, v;
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

/* In mm coordinates */
void
volume_calc_grad (Volume* vout, Volume* vref)
{
    int v;
    int i_p, i, i_n, j_p, j, j_n, k_p, k, k_n; /* p is prev, n is next */
    int gi, gj, gk;
    int idx_p, idx_n;
    float *out_img, *ref_img;

    out_img = (float*) vout->img;
    ref_img = (float*) vref->img;

    v = 0;
    for (k = 0; k < vref->dim[2]; k++) {
	k_p = k - 1;
	k_n = k + 1;
	if (k == 0) k_p = 0;
	if (k == vref->dim[2]-1) k_n = vref->dim[2]-1;
	for (j = 0; j < vref->dim[1]; j++) {
	    j_p = j - 1;
	    j_n = j + 1;
	    if (j == 0) j_p = 0;
	    if (j == vref->dim[1]-1) j_n = vref->dim[1]-1;
	    for (i = 0; i < vref->dim[0]; i++, v++) {
		i_p = i - 1;
		i_n = i + 1;
		if (i == 0) i_p = 0;
		if (i == vref->dim[0]-1) i_n = vref->dim[0]-1;
		
		gi = 3 * v + 0;
		gj = 3 * v + 1;
		gk = 3 * v + 2;
		
		idx_p = volume_index (vref->dim, i_p, j, k);
		idx_n = volume_index (vref->dim, i_n, j, k);
		out_img[gi] = (float) (ref_img[idx_n] - ref_img[idx_p]) / 2.0 / vref->pix_spacing[0];
		idx_p = volume_index (vref->dim, i, j_p, k);
		idx_n = volume_index (vref->dim, i, j_n, k);
		out_img[gj] = (float) (ref_img[idx_n] - ref_img[idx_p]) / 2.0 / vref->pix_spacing[1];
		idx_p = volume_index (vref->dim, i, j, k_p);
		idx_n = volume_index (vref->dim, i, j, k_n);
		out_img[gk] = (float) (ref_img[idx_n] - ref_img[idx_p]) / 2.0 / vref->pix_spacing[2];
	    }
	}
    }
}

Volume* 
volume_make_gradient (Volume* ref)
{
    Volume *grad;
    grad = volume_create (ref->dim, ref->offset, ref->pix_spacing, 
			  PT_VF_FLOAT_INTERLEAVED, 0);
    volume_calc_grad (grad, ref);

    return grad;
}

Volume*
volume_warp (Volume *vout, Volume *vin, Volume *vf)
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
	    return 0;
	}
	if (vout->pix_spacing[d] != vf->pix_spacing[d]) {
	    return 0;
	}
	if (vout->offset[d] != vf->offset[d]) {
	    return 0;
	}
    }

    for (v = 0, k = 0, fz = vf->offset[2]; k < vf->dim[2]; k++, fz += vf->pix_spacing[2]) {
	for (j = 0, fy = vf->offset[1]; j < vf->dim[1]; j++, fy += vf->pix_spacing[1]) {
	    for (i = 0, fx = vf->offset[0]; i < vf->dim[0]; i++, fx += vf->pix_spacing[0], v++) {
		float *dxyz = &vf_img[3*v];
		mz = fz + dxyz[2];
		mk = round_int((mz - vin->offset[2]) / vin->pix_spacing[2]);
		if (mk < 0 || mk >= vin->dim[2]) continue;
		my = fy + dxyz[1];
		mj = round_int((my - vin->offset[1]) / vin->pix_spacing[1]);
		if (mj < 0 || mj >= vin->dim[1]) continue;
		mx = fx + dxyz[0];
		mi = round_int((mx - vin->offset[0]) / vin->pix_spacing[0]);
		if (mi < 0 || mi >= vin->dim[0]) continue;
		mv = (mk * vin->dim[1] + mj) * vin->dim[0] + mi;

#if defined (commentout)
		if (k == 45 && j == 46 && (i == 58 || i == 59)) {
		    printf ("(%d %d %d) (%g %g %g) + (%g %g %g) = (%g %g %g) (%d %d %d)\n",
			i,j,k, fx, fy, fz, dxyz[0], dxyz[1], dxyz[2], 
			mx, my, mz, mi, mj, mk);
		}
#endif
		vout_img[v] = vin_img[mv];
	    }
	}
    }
    return vout;
}

// Computes the intensity differences between two images
Volume*
volume_difference (Volume* vol, Volume* warped)
{
    int i, j, k;
    int p = 0; // Voxel index
    short* temp2;
    short* temp1;
    short* temp3;
    Volume* temp;

    temp = (Volume*) malloc (sizeof(Volume));
    if (!temp) {
	fprintf (stderr, "Memory allocation failed.\n");
	exit(1);
    }

    if(!temp){
	printf("Memory allocation failed for volume...Exiting\n");
	exit(1);
    }
	
    for(i=0;i<3; i++){
	temp->dim[i] = vol->dim[i];
	temp->offset[i] = vol->offset[i];
	temp->pix_spacing[i] = vol->pix_spacing[i];
    }

    temp->npix = vol->npix;
    temp->pix_type = vol->pix_type;
    temp->xmax = vol->xmax;
    temp->xmin = vol->xmin;
    temp->ymax = vol->ymax;
    temp->ymin = vol->ymin;
    temp->zmax = vol->zmax;
    temp->zmin = vol->zmin;

    temp->img = (void*) malloc (sizeof(short)*temp->npix);
    if (!temp->img) {
	fprintf (stderr, "Memory allocation failed.\n");
	exit(1);
    }
    memset (temp->img, -1200, sizeof(short)*temp->npix);

    p = 0; // Voxel index
    temp2 = (short*)vol->img;
    temp1 = (short*)warped->img;
    temp3 = (short*)temp->img;

    for (i=0; i < vol->dim[2]; i++) {
	for (j=0; j < vol->dim[1]; j++) {
	    for (k=0; k < vol->dim[0]; k++) {
		temp3[p] = (temp2[p] - temp1[p]) - 1200;
		p++;
	    }
	}
    }
    return temp;
}

void
vf_convolve_x (Volume* vf_out, Volume* vf_in, float* ker, int width)
{
    int v,x,y,z;
    int half_width;
    float *in_img = (float*) vf_in->img;
    float *out_img = (float*) vf_out->img;

    half_width = width / 2;

    for (v = 0, z = 0; z < vf_in->dim[2]; z++) {
	for (y = 0; y < vf_in->dim[1]; y++) {
	    for (x = 0; x < vf_in->dim[0]; x++, v++) {
		int i, i1;	    /* i is the offset in the vf */
		int j, j1, j2;	    /* j is the index of the kernel */
		int d;		    /* d is the vector field direction */
		float *vin = &in_img[3*v];
		float *vout = &out_img[3*v];

		j1 = x - half_width;
		j2 = x + half_width;
		if (j1 < 0) j1 = 0;
		if (j2 >= vf_in->dim[0]) {
		    j2 = vf_in->dim[0] - 1;
		}
		i1 = j1 - x;
		j1 = j1 - x + half_width;
		j2 = j2 - x + half_width;

		for (d = 0; d < 3; d++) {
		    vout[d] = (float) 0.0;
		    for (i = i1, j = j1; j <= j2; i++, j++) {
			vout[d] += ker[j] * (vin+i*3)[d];
		    }
		}
	    }
	}
    }
}

void
vf_convolve_y (Volume* vf_out, Volume* vf_in, float* ker, int width)
{
    int v,x,y,z;
    int half_width;
    float *in_img = (float*) vf_in->img;
    float *out_img = (float*) vf_out->img;

    half_width = width / 2;

    for (v = 0, z = 0; z < vf_in->dim[2]; z++) {
	for (y = 0; y < vf_in->dim[1]; y++) {
	    for (x = 0; x < vf_in->dim[0]; x++, v++) {
		int i, i1;	    /* i is the offset in the vf */
		int j, j1, j2;	    /* j is the index of the kernel */
		int d;		    /* d is the vector field direction */
		float *vin = &in_img[3*v];
		float *vout = &out_img[3*v];

		j1 = y - half_width;
		j2 = y + half_width;
		if (j1 < 0) j1 = 0;
		if (j2 >= vf_in->dim[1]) {
		    j2 = vf_in->dim[1] - 1;
		}
		i1 = j1 - y;
		j1 = j1 - y + half_width;
		j2 = j2 - y + half_width;

		for (d = 0; d < 3; d++) {
		    vout[d] = (float) 0.0;
		    for (i = i1, j = j1; j <= j2; i++, j++) {
			vout[d] += ker[j] * (vin+i*vf_in->dim[0]*3)[d];
		    }
		}
	    }
	}
    }
}

void
vf_convolve_z (Volume* vf_out, Volume* vf_in, float* ker, int width)
{
    int v,x,y,z;
    int half_width;
    float *in_img = (float*) vf_in->img;
    float *out_img = (float*) vf_out->img;

    half_width = width / 2;

    for (v = 0, z = 0; z < vf_in->dim[2]; z++) {
	for (y = 0; y < vf_in->dim[1]; y++) {
	    for (x = 0; x < vf_in->dim[0]; x++, v++) {
		int i, i1;	    /* i is the offset in the vf */
		int j, j1, j2;	    /* j is the index of the kernel */
		int d;		    /* d is the vector field direction */
		float *vin = &in_img[3*v];
		float *vout = &out_img[3*v];

		j1 = z - half_width;
		j2 = z + half_width;
		if (j1 < 0) j1 = 0;
		if (j2 >= vf_in->dim[2]) {
		    j2 = vf_in->dim[2] - 1;
		}
		i1 = j1 - z;
		j1 = j1 - z + half_width;
		j2 = j2 - z + half_width;

		for (d = 0; d < 3; d++) {
		    vout[d] = (float) 0.0;
		    for (i = i1, j = j1; j <= j2; i++, j++) {
			vout[d] += ker[j] * (vin+i*vf_in->dim[0]*vf_in->dim[1]*3)[d];
		    }
		}
	    }
	}
    }
}
