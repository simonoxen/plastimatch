#define DRR_HUGE_DOUBLE 1e10
#define DRR_LEN_TOLERANCE 1e-6
#define DRR_STRIDE_TOLERANCE 1e-10

#define OUTPUT_FORMAT_PFM 0
#define OUTPUT_FORMAT_PGM 1
#define OUTPUT_FORMAT_RAW 2

/* Returns integer data type */
#define ROUND_INT(x) (((x) >= 0) ? ((int)((x)+0.5)) : (int)(-(-(x)+0.5)))

/* Returns double data type  */
#define ROUND(x) ((float) (ROUND_INT(x)))

/* Returns +1 or -1, depeding on sign.  Zero yeilds +1. */
#define SIGN(x) (((x) >= 0) ? (+1) : (-1))

enum point_location {
    POINTLOC_LEFT,
    POINTLOC_INSIDE,
    POINTLOC_RIGHT,
};
typedef enum point_location Point_location;

struct volume_limit {
    /* upper and lower limits of volume, including tolerances */
    float lower_limit[3];
    float upper_limit[3];

    /* dir == 0 if lower_limit corresponds to lower index */
    int dir[3];
};
typedef struct volume_limit Volume_limit;


/*
 * Summary: Adds the values of the second vector to the the values of the first
 * Parameters: Input is expected to be pointers to two float vectors
 * Return: (void)
 */
void vec3_add2 (
    __local float *v1,
    __local const float *v2
){
    v1[0] += v2[0];
    v1[1] += v2[1];
    v1[2] += v2[2];
}

/*
 * Summary: Adds the values of the second and third vectors and sets those values to be the first vector values
 * Parameters: Input is expected to be pointers to three float vectors
 * Return: (void)
 */
void vec3_add3 (
    __local float *v1,
    __local const float *v2,
    __local const float *v3
){
    v1[0] = v2[0] + v3[0];
    v1[1] = v2[1] + v3[1];
    v1[2] = v2[2] + v3[2];
}


/*
 * Summary: Sets the values of the second vector to the the values of the first
 * Parameters: Input is expected to be pointers to two float vectors
 * Return: (void)
 */
void vec3_copy (
    __local float *v1,
    __local const float *v2
){
    v1[0] = v2[0];
    v1[1] = v2[1];
    v1[2] = v2[2];
}


/*
 * Summary: Computes the dot product of the two vectors
 * Parameters: Input is expected to be pointers to two float vectors
 * Return: Float value which is the dot product of the two vectors
 */
float vec3_dot (
    __local const float *v1,
    __local const float *v2
){
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

/*
 * Summary: Multiplies the vector values by a float value in order to scale the vector
 * Parameters: Input is expected to be a pointer to a float vector, and a float value weight
 * Return: (void)
 */
void vec3_scale2 (
    __local float *v1,
    __local float a
){
    v1[0] *= a;
    v1[1] *= a;
    v1[2] *= a;
}

/*
 * Summary: Multiplies vector values by a float value in order to scale the vector, then sets the weighted vector to replace the values of the first vector
 * Parameters: Input is expected to be two pointers to float vectors, as well as a float value weight
 * Return: (void)
 */
void vec3_scale3 (
    __local float *v1,
    __local const float *v2,
    __local float a
){
    v1[0] = a * v2[0];
    v1[1] = a * v2[1];
    v1[2] = a * v2[2];
}

/*
 * Summary: Subtracts the values of the second from the first and sets those values to be the first vector values
 * Parameters: Input is expected to be pointers to two float vectors
 * Return: (void)
 */
void vec3_sub2 (
    __local float *v1,
    __local const float *v2
){
    v1[0] -= v2[0];
    v1[1] -= v2[1];
    v1[2] -= v2[2];
}

/*
 * Summary: Subtracts the values of the third vector from the second and sets those values to be the first vector values
 * Parameters: Input is expected to be pointers to three float vectors
 * Return: (void)
 */
void vec3_sub3 (
    __local float *v1,
    __local const float *v2,
    __local const float *v3
){
    v1[0] = v2[0] - v3[0];
    v1[1] = v2[1] - v3[1];
    v1[2] = v2[2] - v3[2];
}


/*
 * Summary: Computes the euclidean distance, or length of the vector, from the origin to the head of the vector
 * Parameters: Input is expected to be a pointer to a float vector
 * Return: Float value which is the length of the vector
 */
float vec3_len (
    __local const float *v1
){
    return sqrt(vec3_dot(v1, v1));
}

/*
 * Summary: Normalizes the vector based on the length
 * Parameters: Input is expected to be a pointer to a float vector
 * Return: (void)
 */
void vec3_normalize (
    __local float *v1
){
    vec3_scale2(v1, 1 / vec3_len(v1));
}


/*
 * Summary: Computes the distance between two vectors
 * Parameters: Input is expected to be a pointer to two float vectors
 * Return: (void)
 */
float vec3_dist (
    __local const float *v1,
    __local const float *v2
){
    __local float tmp[3];
    vec3_sub3 (tmp, v1, v2);
    return vec3_len(tmp);
}

/*
 * Summary: Attenuation computation for hu unit
 * Parameters: Input expected to be a float value of the pixel density
 * Return: Float value
 */
float attenuation_lookup_hu (
    float pix_density
){
    if (pix_density <= -800.0) {
	return 0.0;
    } else {
	return pix_density * 0.000022 + 0.022;
    }
}


/*
 * Summary: Kernel function to calculate parameters needed for attenuation
 * Parameters: Input is expected to be a float pointed to an image and an integer value of number of voxels
 * Return: (void)
 */
__kernel void preprocess_attenuation_cl (
    __global float *img,
    int nvoxels
){
    int p = get_global_id(0);

    if (p < nvoxels) {
	img[p] = attenuation_lookup_hu(img[p]);
    }
}

/*
 * Summary: Tests to see where the limits of the boundary for the drr image lies
 * Parameters: Input is expected to be a constant pointer of type Volume_limit and a float value which gagues the volume limit
 * Return: Point_location, as defined in the drr_opencl.h header file
 */
Point_location test_boundary (
    __constant Volume_limit* vol_limit,
    __local float x,
    int i)
{
    if (x < vol_limit->lower_limit[i]) {
	return POINTLOC_LEFT;
    } else if (x > vol_limit->upper_limit[i]) {
	return POINTLOC_RIGHT;
    } else {
	return POINTLOC_INSIDE;
    }
}

/*
 * Summary: Initializes the parameters needed for drr_trace_init
 * Parameters: Input is expected to be the initializing values such as offset
 * Return: (void)
 */
void drr_trace_init_loopvars_nointerp (
    __local int *ai,
    __local int *aidir,
    __local float *ao,
    __local float *al,
    float pt,
    float ry,
    float offset,
    float samp
){
    *aidir = SIGN(ry) * SIGN(samp);
    *ai = ROUND_INT((pt - offset) / samp);
    *ao = SIGN(ry) * (((*ai) * samp + offset) + (SIGN(ry) * 0.5 * fabs(samp)) - pt);

    if (fabs(ry) > DRR_STRIDE_TOLERANCE) {
	*ao = *ao / fabs(ry);
	*al = fabs(samp) / fabs(ry);
    } else {
	*ao = DRR_HUGE_DOUBLE;
	*al = DRR_HUGE_DOUBLE;
    }
}

/*
 * Summary: Initializes the DRR trace
 * Parameters: Input is the volume parameters and the x,y,z dimension pointers
 * Return: int value of 0 or 1, depending on intersection
 */
int drr_trace_init (
    __local int *ai_x,
    __local int *ai_y,
    __local int *ai_z,
    __local int *aixdir,
    __local int *aiydir,
    __local int *aizdir,
    __local float *ao_x,
    __local float *ao_y,
    __local float *ao_z,
    __local float *al_x,
    __local float *al_y,
    __local float *al_z,
    __local float *len,
    __constant float *offset,
    __constant float *pix_spacing,
    __constant Volume_limit *vol_limit,
    __local float *p1,
    __local float *p2
){
    Point_location ploc[3][2];
    float alpha_lo[3], alpha_hi[3];
    float alpha_in, alpha_out;
    __local float ray[3];
    float tmp;
	
    /* Compute the ray */
    vec3_sub3(ray, p2, p1);
	
    for (int d = 0; d < 3; d++) {
	ploc[d][0] = test_boundary (vol_limit, p1[d], d);
	ploc[d][1] = test_boundary (vol_limit, p2[d], d);
	/* Immediately reject segments which don't intersect the volume in 
	   this dimension */
	if (ploc[d][0] == POINTLOC_LEFT && ploc[d][1] == POINTLOC_LEFT) {
	    return 0;
	}
	if (ploc[d][0] == POINTLOC_RIGHT && ploc[d][1] == POINTLOC_RIGHT) {
	    return 0;
	}
    }

    /* If we made it here, all three dimensions have some range of alpha
       where they intersects the volume.  However, these alphas might 
       not overlap.  We compute the alphas, then test overlapping 
       alphas to find the segment range within the volume.  */
    for (int d = 0; d < 3; d++) {
	/* If ray is parallel to grid, location must be inside */
	if (fabs(ray[d]) < DRR_LEN_TOLERANCE) {
	    if (ploc[d][0] != POINTLOC_INSIDE) {
		return 0;
	    }
	    alpha_lo[d] = - DBL_MAX;
	    alpha_hi[d] = + DBL_MAX;
	} else {
	    alpha_lo[d] = (vol_limit->lower_limit[d] - p1[d]) / ray[d];
	    alpha_hi[d] = (vol_limit->upper_limit[d] - p1[d]) / ray[d];

	    /* Sort alphas */
	    if (alpha_hi[d] < alpha_lo[d]) {
		tmp = alpha_hi[d];
		alpha_hi[d] = alpha_lo[d];
		alpha_lo[d] = tmp;
	    }

	    /* Clip alphas to segment */
	    if (alpha_lo[d] < 0.0)
		alpha_lo[d] = 0.0;
	    if (alpha_lo[d] > 1.0)
		alpha_lo[d] = 1.0;
	    if (alpha_hi[d] < 0.0)
		alpha_hi[d] = 0.0;
	    if (alpha_hi[d] > 1.0)
		alpha_hi[d] = 1.0;
	}
    }
	
    /* alpha_in is the alpha where the segment enters the boundary, and 
       alpha_out is where it exits the boundary.  */
    alpha_in = alpha_lo[0];
    alpha_out = alpha_hi[0];
    for (int d = 1; d < 3; d++) {
	if (alpha_in < alpha_lo[d])
	    alpha_in = alpha_lo[d];
	if (alpha_out > alpha_hi[d])
	    alpha_out = alpha_hi[d];
    }

    /* If exit is before entrance, the segment does not intersect the volume */
    if (alpha_out - alpha_in < DRR_LEN_TOLERANCE) {
	return 0;
    }

    /* Create the volume intersection points */
    for (int d = 0; d < 3; d++) {
	ips[0][d] = p1[d] + alpha_in * ray[d];
	ips[1][d] = p1[d] + alpha_out * ray[d];
    }

    /* Create the volume intersection points */
    vec3_sub3(ray, p2, p1);
    vec3_normalize(ray);

    /* We'll go from p1 to p2 */
    /* Variable notation:
       ai_x	// index of x
       aixdir  // x indices moving up or down?
       ao_x	// absolute length to next voxel crossing
       al_x	// length between voxel crossings
    */
    drr_trace_init_loopvars_nointerp(ai_x, aixdir, ao_x, al_x, ips[0][0], ray[0], offset[0], pix_spacing[0]);
    drr_trace_init_loopvars_nointerp(ai_y, aiydir, ao_y, al_y, ips[0][1], ray[1], offset[1], pix_spacing[1]);
    drr_trace_init_loopvars_nointerp(ai_z, aizdir, ao_z, al_z, ips[0][2], ray[2], offset[2], pix_spacing[2]);
	
    *len = vec3_dist(&ips[0][0], &ips[1][0]);

    return 1;
}


/*
 * Summary: DRR Trace without interpolation
 * Parameters: Volume parameters
 * Return: float of the accumulation value
 */
float drr_trace_ray_nointerp_2009 (
    __global float *dev_vol,
    __constant int *vol_dim,
    __constant float *offset,
    __constant float *pix_spacing,
    __constant Volume_limit *vol_limits,
    __constant float *p1in,
    __local float *p2in,
    int msd_fp,
    int preprocess_attenuation,
    long img_idx
){
    __local int ai_x, ai_y, ai_z;
    __local int aixdir, aiydir, aizdir;
    __local float ao_x, ao_y, ao_z;
    __local float al_x, al_y, al_z;
    __local float len;
    float aggr_len = 0.0;
    float accum = 0.0;
    float msd_bins[DRR_MSD_NUM_BINS];
	
    if (!drr_trace_init(&ai_x, &ai_y, &ai_z, &aixdir, &aiydir, &aizdir, &ao_x, &ao_y, &ao_z, &al_x, &al_y, &al_z, &len, offset, pix_spacing, vol_limits, p1in, p2in)) {
	return 0.0;
    }

    /* We'll go from p1 to p2 */
    /* Variable notation:
       ai_x	// index of x
       aixdir  // x indices moving up or down?
       ao_x	// absolute length to next voxel crossing
       al_x	// length between voxel crossings
    */

    long x;
    float pix_density;
    float pix_len;

    do {
	x = ai_z * vol_dim[0] * vol_dim[1] + ai_y * vol_dim[0] + ai_x;
		
	if (ai_z < vol_dim[2] && ai_z > 0 && ai_y < vol_dim[1] && ai_y > 0 && ai_x < vol_dim[0] && ai_x > 0)
	    pix_density = dev_vol[x];
	else
	    pix_density = 0.0;
		
	if ((ao_x < ao_y) && (ao_x < ao_z)) {
	    pix_len = ao_x;
	    aggr_len += ao_x;
	    ao_y -= ao_x;
	    ao_z -= ao_x;
	    ao_x = al_x;
	    ai_x += aixdir;
	} else if (ao_y < ao_z) {
	    pix_len = ao_y;
	    aggr_len += ao_y;
	    ao_x -= ao_y;
	    ao_z -= ao_y;
	    ao_y = al_y;
	    ai_y += aiydir;
	} else {
	    pix_len = ao_z;
	    aggr_len += ao_z;
	    ao_x -= ao_z;
	    ao_y -= ao_z;
	    ao_z = al_z;
	    ai_z += aizdir;
	}

	if (preprocess_attenuation)
	    accum += pix_len * pix_density;
	else
	    accum += pix_len * attenuation_lookup_hu(pix_density);

    } while (aggr_len + DRR_LEN_TOLERANCE < len);

    return accum;
}

/*
 * Summary: DRR Kernel
 * Parameters: Volume parameters
 * Return: (void)
 */
__kernel void kernel_drr (
    __global float *dev_vol,
    __global float *dev_img,
    __constant int *vol_dim,
    __constant int2 *img_dim,
    __constant float *offset, 
    __constant float *pix_spacing,
    __constant Volume_limit *vol_limits,
    __constant float *p1,
    __constant float *ul_room,
    __constant float *incr_r,
    __constant float *incr_c,
    __constant int4 *ndevice,
    float scale,
    int output_format,
    int msd_fp,
    int preprocess_attenuation,
    int exponential_mapping,
    int pixel_offset
){
    uint c = get_global_id(0);
    uint r = get_global_id(1);

    if (c >= (*ndevice).x || r >= (*ndevice).y)
	return;

    // Index row major into the image
    long img_idx = c + (r * (*img_dim).x);
    r += pixel_offset;

    float r_tgt[3];
    r_tgt[0] = ul_room[0] + (incr_r[0] * r);
    r_tgt[1] = ul_room[1] + (incr_r[1] * r);
    r_tgt[2] = ul_room[2] + (incr_r[2] * r);

    __local float p2[3];
    p2[0] = r_tgt[0] + incr_c[0] * c;
    p2[1] = r_tgt[1] + incr_c[1] * c;
    p2[2] = r_tgt[2] + incr_c[2] * c;

    float value = drr_trace_ray_nointerp_2009(dev_vol, vol_dim, offset, pix_spacing, lower_limits, p1, p2, msd_fp, preprocess_attenuation, img_idx);

    // Translate from pixels to cm*gm
    value *= 0.1;
	
    if (exponential_mapping) {
	value = exp(-value);
    }

    // User requested scaling
    value *= scale;

    dev_img[img_idx] = value;
}
