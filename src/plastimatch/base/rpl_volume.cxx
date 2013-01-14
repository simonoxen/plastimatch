/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler_warnings.h"
#include "logfile.h"
#include "mha_io.h"
#include "plm_int.h"
#include "plm_math.h"
#include "proj_matrix.h"
#include "proj_volume.h"
#include "ray_trace.h"
#include "rpl_volume.h"
#include "volume.h"
#include "volume_limit.h"
#include "volume_macros.h"
#include "print_and_exit.h"

//#define VERBOSE 1

class Ray_data {
public:
    int ap_idx;
    bool intersects_volume;
    double ip1[3];       /* Front intersection with volume */
    double ip2[3];       /* Back intersection with volume */
    double p2[3];        /* Intersection with aperture plane */
    double ray[3];       /* Unit vector in direction of ray */
    double front_dist;   /* Distance from aperture to ip1 */
    double back_dist;    /* Distance from aperture to ip2 */
    double cp[3];        /* Intersection with front clipping plane */
};

typedef struct callback_data Callback_data;
struct callback_data {
    Rpl_volume *rpl_vol;    /* Radiographic depth volume */
    Ray_data *ray_data;
    int* ires;              /* Aperture Dimensions */
    int step_offset;        /* Number of steps before first ray sample */
    double accum;           /* Accumulated intensity */
};

class Rpl_volume_private {
public:
    Proj_volume *proj_vol;
    Ray_data *ray_data;
    double front_clipping_dist;
    double back_clipping_dist;

public:
    Rpl_volume_private () {
        proj_vol = new Proj_volume;
        ray_data = 0;
        front_clipping_dist = DBL_MAX;
        back_clipping_dist = -DBL_MAX;
    }
    ~Rpl_volume_private () {
        delete proj_vol;
        if (ray_data) {
            delete[] ray_data;
        }
    }
};

Rpl_volume::Rpl_volume () {
    d_ptr = new Rpl_volume_private;
}

Rpl_volume::~Rpl_volume () {
    delete d_ptr;
}

void 
Rpl_volume::set_geometry (
    const double src[3],           // position of source (mm)
    const double iso[3],           // position of isocenter (mm)
    const double vup[3],           // dir to "top" of projection plane
    double sid,                    // dist from proj plane to source (mm)
    const int image_dim[2],        // resolution of image
    const double image_center[2],  // image center (pixels)
    const double image_spacing[2], // pixel size (mm)
    const double step_length       // spacing between planes
)
{
    double clipping_dist[2] = {sid, sid};

#if defined (commentout)
    printf ("> src = %f %f %f\n", src[0], src[1], src[2]);
    printf ("> iso = %f %f %f\n", iso[0], iso[1], iso[2]);
    printf ("> vup = %f %f %f\n", vup[0], vup[1], vup[2]);
    printf ("> sid = %f\n", sid);
    printf ("> idim = %d %d\n", image_dim[0], image_dim[1]);
    printf ("> ictr = %f %f\n", image_center[0], image_center[1]);
    printf ("> isp = %f %f\n", image_spacing[0], image_spacing[1]);
    printf ("> stp = %f\n", step_length);
#endif

    /* This sets everything except the clipping planes.  We don't know 
       these until caller tells us the CT volume to compute against. */
    d_ptr->proj_vol->set_geometry (
        src, iso, vup, sid, image_dim, image_center, image_spacing,
        clipping_dist, step_length);
}

static double
lookup_rgdepth (
    Rpl_volume *rpl_vol, 
    int ap_ij[2], 
    double dist
)
{
    plm_long idx1, idx2;
    plm_long ijk[3];
    double rg1, rg2, rgdepth, frac;
    Proj_volume *proj_vol = rpl_vol->get_proj_volume ();
    Volume *vol = rpl_vol->get_volume();
    float* d_img = (float*) vol->img;

    if (dist < 0) {
        return 0.0;
    }

    ijk[0] = ap_ij[0];
    ijk[1] = ap_ij[1];
    ijk[2] = (int) floorf (dist / proj_vol->get_step_length());
    /* Depth to step before point */
    idx1 = volume_index (vol->dim, ijk);
    if (idx1 < vol->npix) {
        rg1 = d_img[idx1];
    } else {
        return 0.0f;
    }

    /* Fraction from step before point to point */
    frac = (dist - ijk[2] * proj_vol->get_step_length()) 
        / proj_vol->get_step_length();
    
#if defined (commentout)
    printf ("(%g - %d * %g) / %g = %g\n", dist, ijk[2], rpl_vol->ray_step, 
	rpl_vol->ray_step, frac);
#endif

    /* Depth to step after point */
    ijk[2]++;
    idx2 = volume_index (vol->dim, ijk);
    if (idx2 < vol->npix) {
        rg2 = d_img[idx2];
    } else {
        rg2 = d_img[idx1];
    }

    /* Radiographic depth, interpolated in depth only */
    rgdepth = rg1 + frac * (rg2 - rg1);

    return rgdepth;
}

/* Lookup radiological path length to a voxel in world space */
double
Rpl_volume::get_rgdepth (
    const double* ct_xyz         /* I: location of voxel in world space */
)
{
    int ap_ij[2], ap_idx;
    double ap_xy[3];
    double dist, rgdepth = 0.;
    int debug = 0;

    /* For debugging */
    if ((ct_xyz[0] > -198 && ct_xyz[0] < -196)
	&& (ct_xyz[1] > 132 && ct_xyz[1] < 134)
	&& (ct_xyz[2] > -6 && ct_xyz[2] < 6))
    {
	debug = 1;
    }
#if defined (commentout)
#endif

    /* A couple of abbreviations */
    const int *ires = d_ptr->proj_vol->get_image_dim();
    Proj_matrix *pmat = d_ptr->proj_vol->get_proj_matrix();

    /* Back project the voxel to the aperture plane */
    mat43_mult_vec3 (ap_xy, pmat->matrix, ct_xyz);
    ap_xy[0] = pmat->ic[0] + ap_xy[0] / ap_xy[2];
    ap_xy[1] = pmat->ic[1] + ap_xy[1] / ap_xy[2];

    /* Make sure value is not inf or NaN */
    if (!is_number (ap_xy[0]) || !is_number (ap_xy[1])) {
    	return -1;
    }

    /* Round to nearest aperture index */
    ap_ij[0] = ROUND_INT (ap_xy[0]);
    ap_ij[1] = ROUND_INT (ap_xy[1]);

    if (debug) {
	printf ("ap_xy = %g %g\n", ap_xy[0], ap_xy[1]);
    }

    /* Only handle voxels inside the (square) aperture */
    if (ap_ij[0] < 0 || ap_ij[0] >= ires[0] ||
        ap_ij[1] < 0 || ap_ij[1] >= ires[1]) {
        return -1;
    }

    ap_idx = ap_ij[1] * ires[0] + ap_ij[0];

    /* Look up pre-computed data for this ray */
    Ray_data *ray_data = &d_ptr->ray_data[ap_idx];
    double *ap_xyz = ray_data->p2;

    if (debug) {
	printf ("ap_xyz = %g %g %g\n", ap_xyz[0], ap_xyz[1], ap_xyz[2]);
    }

    /* Compute distance from aperture to voxel */
    dist = vec3_dist (ap_xyz, ct_xyz);

    /* Retrieve the radiographic depth */
    rgdepth = lookup_rgdepth (this, ap_ij, dist);

    if (debug) {
	printf ("(%g %g %g / %g %g %g) -> (%d %d %g) -> %g\n", 
	    ct_xyz[0], ct_xyz[1], ct_xyz[2], 
	    (ct_xyz[0] + 249) / 2,
	    (ct_xyz[1] + 249) / 2,
	    (ct_xyz[2] + 249) / 2,
	    ap_ij[0], ap_ij[1], dist, 
	    rgdepth);
    }

    return rgdepth;
}

void 
Rpl_volume::compute (Volume *ct_vol)
{
    int ires[2];
    Volume_limit ct_limit;

    /* A couple of abbreviations */
    Proj_volume *proj_vol = d_ptr->proj_vol;
    const double *src = proj_vol->get_src();
    const double *nrm = proj_vol->get_nrm();
    ires[0] = d_ptr->proj_vol->get_image_dim (0);
    ires[1] = d_ptr->proj_vol->get_image_dim (1);

    /* Compute volume boundary box */
    volume_limit_set (&ct_limit, ct_vol);

    proj_vol->debug ();

    /* Make two passes through the aperture grid.  The first pass 
       is used to find the clipping planes.  The second pass actually 
       traces the rays. */

    /* Allocate data for each ray */
    if (d_ptr->ray_data) delete[] d_ptr->ray_data;
    d_ptr->ray_data = new Ray_data[ires[0]*ires[1]];

    /* Scan through the aperture -- first pass */
    for (int r = 0; r < ires[0]; r++) {
        double r_tgt[3];
        double tmp[3];

        /* Compute r_tgt = 3d coordinates of first pixel in this row
           on aperture */
        vec3_copy (r_tgt, proj_vol->get_ul_room());
        vec3_scale3 (tmp, proj_vol->get_incr_r(), (double) r);
        vec3_add2 (r_tgt, tmp);

        for (int c = 0; c < ires[1]; c++) {
            /* Compute index of aperture pixel */
            plm_long ap_idx = r * ires[0] + c;

            /* Make some aliases */
            Ray_data *ray_data = &d_ptr->ray_data[ap_idx];
            double *ip1 = ray_data->ip1;
            double *ip2 = ray_data->ip2;
            double *p2 = ray_data->p2;
            double *ray = ray_data->ray;

            /* Save the aperture index */
            ray_data->ap_idx = ap_idx;

            /* Compute p2 = 3d coordinates of point on aperture */
            vec3_scale3 (tmp, proj_vol->get_incr_c(), (double) c);
            vec3_add3 (p2, r_tgt, tmp);

	    /* Define unit vector in ray direction */
	    vec3_sub3 (ray, p2, src);
	    vec3_normalize1 (ray);

	    /* Test if ray intersects volume and create intersection points */
            ray_data->intersects_volume = false;
	    if (!volume_limit_clip_ray (&ct_limit, ip1, ip2, src, ray)) {
		continue;
	    }
            
	    /* If intersect points are before or after aperture. 
               If before, clip them at aperture plane. */

            /* First, check the second point */
            double tmp[3];
            vec3_sub3 (tmp, ip2, p2);
            if (vec3_dot (tmp, nrm) > 0) {
                /* If second point is behind aperture, then so is 
                   first point, and therefore the ray doesn't intersect 
                   the volume. */
                continue;
            }

            /* OK, by now we know this ray does intersect the volume */
            ray_data->intersects_volume = true;

#if defined (commentout)
	    printf ("(%d,%d)\n", r, c);
	    printf ("ap  = %f %f %f\n", p2[0], p2[1], p2[2]);
	    printf ("ip1 = %f %f %f\n", ip1[0], ip1[1], ip1[2]);
	    printf ("ip2 = %f %f %f\n", ip2[0], ip2[1], ip2[2]);
#endif

            /* Compute distance to front intersection point, and set 
               front clipping plane if indicated */
            vec3_sub3 (tmp, ip1, p2);
            if (vec3_dot (tmp, nrm) > 0) {
                ray_data->front_dist = 0;
            } else {
                ray_data->front_dist = vec3_dist (p2, ip1);
            }
            if (ray_data->front_dist < d_ptr->front_clipping_dist) {
                d_ptr->front_clipping_dist = ray_data->front_dist;
            }

            /* Compute distance to back intersection point, and set 
               back clipping plane if indicated */
	    ray_data->back_dist = vec3_dist (p2, ip2);
            if (ray_data->back_dist > d_ptr->back_clipping_dist) {
                d_ptr->back_clipping_dist = ray_data->back_dist;
            }
#if defined (commentout)
	    printf ("fd/bd = %f %f\n", ray_data->front_dist,
                ray_data->back_dist);
#endif
        }
    }

    if (d_ptr->front_clipping_dist == DBL_MAX) {
        print_and_exit ("Sorry, total failure intersecting volume\n");
    }

    lprintf ("FPD = %f, BPD = %f\n", 
        d_ptr->front_clipping_dist, d_ptr->back_clipping_dist);

    /* Ahh.  Now we can set the clipping planes and allocate the 
       actual volume. */
    double clipping_dist[2] = {
        d_ptr->front_clipping_dist, d_ptr->back_clipping_dist};
    d_ptr->proj_vol->set_clipping_dist (clipping_dist);
    d_ptr->proj_vol->allocate ();
    
    /* Scan through the aperture -- second pass */
    for (int r = 0; r < ires[0]; r++) {

        //if (r % 50 == 0) printf ("Row: %4d/%d\n", r, rows);

        for (int c = 0; c < ires[1]; c++) {

            /* Compute index of aperture pixel */
            plm_long ap_idx = r * ires[0] + c;

            /* Make some aliases */
            Ray_data *ray_data = &d_ptr->ray_data[ap_idx];

            /* Compute intersection with front clipping plane */
            vec3_scale3 (ray_data->cp, ray_data->ray, 
                d_ptr->front_clipping_dist);
            vec3_add2 (ray_data->cp, ray_data->p2);

#if defined (commentout)
	    printf ("Tracing ray (%d,%d)\n", r, c);
#endif

            this->ray_trace (
                ct_vol,       /* I: CT volume */
                ray_data,     /* I: Pre-computed data for this ray */
                &ct_limit,    /* I: CT bounding region */
                src,          /* I: @ source */
                ires          /* I: ray cast resolution */
            );

        }
    }
}

void 
Rpl_volume::compute_wed_volume (Volume *wed_vol, Volume *in_vol, float background)
{
  /* A couple of abbreviations */
    Proj_volume *proj_vol = d_ptr->proj_vol;
    Volume *rvol = proj_vol->get_volume();
    float *rvol_img = (float*) rvol->img;
    float *in_vol_img = (float*) in_vol->img;
    float *wed_vol_img = (float*) wed_vol->img;
    const int *ires = proj_vol->get_image_dim();

 
    plm_long wijk[3];  /* Index within wed_volume */


    //   printf("ires is %d %d %d %d \n",ires[0],ires[1],ires[2],ires[3]);
    for (wijk[1] = 0; wijk[1] < ires[1]; wijk[1]++) {

        for (wijk[0] = 0; wijk[0] < ires[0]; wijk[0]++) {
            /* Compute index of aperture pixel */
            plm_long ap_idx = wijk[1] * ires[0] + wijk[0];

            bool debug = false;
            if (ap_idx == (ires[1]/2) * ires[0] + (ires[0] / 2)) {
	      //                printf ("DEBUGGING %d %d\n", ires[1], ires[0]);
	      //                debug = true;
            }
#if defined (commentout)
#endif

            /* Make some aliases */
            Ray_data *ray_data = &d_ptr->ray_data[ap_idx];

	    //Set the default to background, if ray misses volume
	    //ires[2] - is this meaningless?
            if (!ray_data->intersects_volume) {
	      for (wijk[2] = 0; wijk[2] < ires[2]; wijk[2]++) {
                plm_long widx = volume_index (rvol->dim, wijk);
		wed_vol_img[widx] = background;
	      }
                continue;
            }

            /* Index within rpl_volume */
            plm_long rijk[3] = { wijk[0], wijk[1], 0 };

            /* Loop, looking for each output voxel */
            for (wijk[2] = 0; wijk[2] < rvol->dim[2]; wijk[2]++) {
                plm_long widx = volume_index (rvol->dim, wijk);

		//Set the default to background.
		wed_vol_img[widx] = background;

                /* Compute the currently required rpl for this step */
                double req_rpl = wijk[2] * 1.0;

                if (debug) printf ("--- (%d,%f)\n", (int) wijk[2], req_rpl);

                /* Loop through input voxels looking for appropriate 
                   value */
                while (rijk[2] < rvol->dim[2]) {
                    plm_long ridx = volume_index (rvol->dim, rijk);
                    double curr_rpl = rvol_img[ridx];

                    if (debug) printf ("(%d,%f)\n", (int) rijk[2], curr_rpl);

                    /* Test if the current input voxel is suitable */
                    if (curr_rpl > req_rpl) {
                        /* Compute coordinate of matching voxel */
                        double xyz[3];
                        double dist = rijk[2] * proj_vol->get_step_length();
                        vec3_scale3 (xyz, ray_data->ray, dist);
                        vec3_add2 (xyz, ray_data->cp);
                        
                        /* Look up value at coordinate in input image */
                        plm_long in_ijk[3];
                        in_ijk[2] = ROUND_PLM_LONG(
                            (xyz[2] - in_vol->offset[2]) / in_vol->spacing[2]);
                        in_ijk[1] = ROUND_PLM_LONG(
                            (xyz[1] - in_vol->offset[1]) / in_vol->spacing[1]);
                        in_ijk[0] = ROUND_PLM_LONG(
                            (xyz[0] - in_vol->offset[0]) / in_vol->spacing[0]);

                        if (debug) {
                            printf ("%f %f %f\n", xyz[0], xyz[1], xyz[2]);
                            printf ("%d %d %d\n", (int) in_ijk[0], 
                                (int) in_ijk[1], (int) in_ijk[2]);
                        }

                        if (in_ijk[2] < 0 || in_ijk[2] >= in_vol->dim[2])
                            break;
                        if (in_ijk[1] < 0 || in_ijk[1] >= in_vol->dim[1])
                            break;
                        if (in_ijk[0] < 0 || in_ijk[0] >= in_vol->dim[0])
                            break;

                        plm_long in_idx = volume_index(in_vol->dim, in_ijk);

			float value = in_vol_img[in_idx];
			
			/* Write value to output image */
			wed_vol_img[widx] = value;



                        /* Suitable voxel found and processed, so move on 
                           to the next output voxel */
                        break;
                    }
                    /* Otherwise, current voxel has insufficient 
                       rpl, so move on to the next */
                    rijk[2] ++;
                }
            }
        }
    }
}


void 
Rpl_volume::compute_dew_volume (Volume *wed_vol, Volume *dew_vol, float background)
{
  
  double dummy_vec[3] = {0., 0., 0.};
  double dummy_length = 0.;

  double master_coord[2]; //coordinate within a unit box that determines weighting of the final trilinear interpolation
  double master_square[2][2]; //"box" containing the 4 values used for the final bilinear interpolation

  //A couple of abbreviations
  Proj_volume *proj_vol = d_ptr->proj_vol;
  Volume *rvol = proj_vol->get_volume();
  float *rvol_img = (float*) rvol->img;
  float *dew_vol_img = (float*) dew_vol->img;
  float *wed_vol_img = (float*) wed_vol->img;
  const plm_long *dew_dim = dew_vol->dim; 
  
  //Get some parameters from the proj volume
  const int *ires = proj_vol->get_image_dim();

 

  const double *src = proj_vol->get_src();
  const double dist = proj_vol->get_proj_matrix()->sid; //distance from source to aperture
  double src_iso_vec[3];   //vector from source to isocenter
  proj_vol->get_proj_matrix()->get_nrm(src_iso_vec); 
  vec3_invert(src_iso_vec);

  printf("%f %f %f\n",src_iso_vec[0],src_iso_vec[1],src_iso_vec[2]);
  
  //Contruct aperture "box", in which each voxel's respective
  //ray intersection must be within.
  
  Ray_data *ray_box[4];
  ray_box[0] = &d_ptr->ray_data[ 0 ];
  ray_box[1] = &d_ptr->ray_data[ ires[0]-1 ];
  ray_box[2] = &d_ptr->ray_data[ ires[0]*(ires[1]-1) ];
  ray_box[3] = &d_ptr->ray_data[ ires[0]*ires[1]-1 ];

  //Compute aperture dimension lengths and normalized axes
  double ap_dim[2]; //length of each aperture axis
  double ap_axis1[3]; //unit vector of ap. axis 1
  double ap_axis2[3];
  vec3_sub3(ap_axis1,ray_box[1]->p2,ray_box[0]->p2);
  ap_dim[0] = vec3_len(ap_axis1);
  vec3_normalize1(ap_axis1);
  vec3_sub3(ap_axis2,ray_box[2]->p2,ray_box[0]->p2);
  ap_dim[1] = vec3_len(ap_axis2);
  vec3_normalize1(ap_axis2);
  
  Ray_data *ray_adj[4]; //the 4 rays in rpl space that border each coordinate
  double ray_adj_len[4]; //calculated length of adjacent 4 rays to each voxel

  printf("ray 0 is %f %f %f\n",ray_box[0]->p2[0],ray_box[0]->p2[1],ray_box[0]->p2[2]);
  printf("ray 1 is %f %f %f\n",ray_box[1]->p2[0],ray_box[1]->p2[1],ray_box[1]->p2[2]);
  printf("ray 2 is %f %f %f\n",ray_box[2]->p2[0],ray_box[2]->p2[1],ray_box[2]->p2[2]);
  printf("ray 3 is %f %f %f\n",ray_box[3]->p2[0],ray_box[3]->p2[1],ray_box[3]->p2[2]);

  /*
  plm_long wijk[3]; 
  for (wijk[1] = 0; wijk[1] < ires[1]; wijk[1]++) {
    for (wijk[0] = 0; wijk[0] < ires[0]; wijk[0]++) {
      plm_long ap_idx = wijk[1] * ires[0] + wijk[0];
      Ray_data *ray_data = &d_ptr->ray_data[ap_idx];
      printf("ray (%d,%d) is %f %f %f\n",wijk[0],wijk[1],ray_data->p2[0],ray_data->p2[1],ray_data->p2[2]);
    }
  }
  */
  
  plm_long wijk[3]; //index within wed_volume
  plm_long rijk[3]; //index within rpl_volume
  plm_long dijk[3]; //Index within dew_volume
  plm_long didx; //image index within dew_volume

  double coord[3];   //coordinate within dew_volume
  double ap_coord[3]; //coordinate in aperture plane along coord vector
  double ap_coord_plane[2]; //transformed, 2-d aperture coordinate of each voxel ray

  double coord_vec[3]; //vector along source to coordinate
  double unit_coord_vec[3]; //unit vector along source to coordinate
  double ap_coord_vec[3]; //vector along source to coordinate, terminated at ap. plane

  double coord_ap_len; //distance from coordinate to aperture
  double dummy_lin_ex;
  plm_long dummy_index1;
  plm_long dummy_index2;

  plm_long ray_lookup[4][2]; //quick lookup table for ray coordinates for rijk input
  double ray_rad_len[4]; //radiation length of each ray
	       
  for (dijk[0] = 0; dijk[0] != dew_dim[0]; ++dijk[0])  {
    coord[0] = dijk[0]*dew_vol->spacing[0]+dew_vol->offset[0];
    for (dijk[1] = 0; dijk[1] != dew_dim[1]; ++dijk[1])  {
      coord[1] = dijk[1]*dew_vol->spacing[1]+dew_vol->offset[1];
      for (dijk[2] = 0; dijk[2] != dew_dim[2]; ++dijk[2])  {
	coord[2] = dijk[2]*dew_vol->spacing[2]+dew_vol->offset[2];
	
	didx = volume_index (dew_dim, dijk);
	
	//Set the default to background.
	dew_vol_img[didx] = background;

	
	vec3_sub3(coord_vec,coord,src); //Determine the vector to this voxel from the source
	vec3_copy(unit_coord_vec,coord_vec);
	vec3_normalize1(unit_coord_vec); //Get unit vector from source to voxel
	coord_ap_len = dist/vec3_dot(unit_coord_vec,src_iso_vec); //trig + dot product for distance
	vec3_copy(ap_coord_vec,unit_coord_vec);
	vec3_scale2(ap_coord_vec, coord_ap_len); //calculate vector from source to aperture plane
	vec3_add3(ap_coord,ap_coord_vec,src);



	//	printf("coord is %f %f %f at voxel %d\n",ap_coord[0],ap_coord[1],ap_coord[2], didx);


	//As ap_coord is on the proj. plane, then simply check the 6 coord. boundaries
	  if ( (ap_coord[0] <= ray_box[0]->p2[0]) && (ap_coord[0] >= ray_box[3]->p2[0]) &&
	       (ap_coord[1] <= ray_box[0]->p2[1]) && (ap_coord[1] >= ray_box[3]->p2[1]) &&
	       (ap_coord[2] <= ray_box[0]->p2[2]) && (ap_coord[2] >= ray_box[3]->p2[2]) )  {
	    
	    for (int i=0;i!=4;++i)  {master_square[i/2][i%2] = background;}

	    //Now we must find the projection of the point on the two aperture axes
	    //Do this by calculating the closest point along both

	    vec3_sub3(dummy_vec,ap_coord,ray_box[0]->p2);
	    dummy_length = vec3_len(dummy_vec);
	    vec3_normalize1(dummy_vec);
	    ap_coord_plane[0] = vec3_dot(dummy_vec,ap_axis1)*dummy_length;
	    ap_coord_plane[1] = vec3_dot(dummy_vec,ap_axis2)*dummy_length;

	    //Note: starting here, some of the following code implicitly assumes that the 
	    //aperture spacing is 1mm.  If this changes, some normalizing will be needed.
	    master_coord[0] = ap_coord_plane[0]-floor(ap_coord_plane[0]);
	    master_coord[1] = ap_coord_plane[1]-floor(ap_coord_plane[1]);

	    //Get the 4 adjacent rays relative to the aperature coordinates
	    int base_ap_coord = (int) (floor(ap_coord_plane[0]) + floor(ap_coord_plane[1])*ires[0]);
	    ray_adj[0] = &d_ptr->ray_data[ base_ap_coord ];
	    ray_adj[1] = &d_ptr->ray_data[ base_ap_coord + 1 ];
	    ray_adj[2] = &d_ptr->ray_data[ base_ap_coord + ires[0] ];
	    ray_adj[3] = &d_ptr->ray_data[ base_ap_coord + ires[0] + 1 ];

	    //Compute ray indices for later rpl calculations.
	    ray_lookup[0][0] = floor(ap_coord_plane[0]);
	    ray_lookup[0][1] = floor(ap_coord_plane[1]);
	    ray_lookup[1][0] = floor(ap_coord_plane[0]) + 1;
	    ray_lookup[1][1] = floor(ap_coord_plane[1]);
	    ray_lookup[2][0] = floor(ap_coord_plane[0]);
	    ray_lookup[2][1] = floor(ap_coord_plane[1]) + 1;
	    ray_lookup[3][0] = floor(ap_coord_plane[0]) + 1;
	    ray_lookup[3][1] = floor(ap_coord_plane[1]) + 1;

	    //Now compute the distance along each of the 4 rays
	    //Distance chosen to be the intersection of each ray with the plane that both
	    //contains the voxel and is normal to the aperture plane.

	    for (int i=0;i!=4;++i)  {
	      //Compute distance along each ray
	      ray_adj_len[i] = (vec3_len(coord_vec)/coord_ap_len)*vec3_len(ray_adj[i]->p2);
	      //Now look up the radiation length.  Since this is an exact coordinate,
	      //linearly extrapolate along the rpl rays to obtain this value.
	      dummy_lin_ex = ray_adj_len[i]/proj_vol->get_step_length() - floor(ray_adj_len[i]/proj_vol->get_step_length());
	      rijk[0] = ray_lookup[i][0];
	      rijk[1] = ray_lookup[i][1];
	      rijk[2] = (int) floor(ray_adj_len[i]/proj_vol->get_step_length());
	      dummy_index1 = volume_index ( rvol->dim, rijk );
	      rijk[2] = (int) ceil(ray_adj_len[i]/proj_vol->get_step_length());
	      dummy_index2 = volume_index ( rvol->dim, rijk );

	      ray_rad_len[i] = rvol_img[dummy_index1] * (1-dummy_lin_ex) + rvol_img[dummy_index2] * dummy_lin_ex;

	      //DIVERGENCE HERE:
	      //Do we attempt a sort of trilinear interpolation, with a shape that is not regular (z dim different for each ray)?
	      //Or simply linearly interpolate each ray to get a dose value, then binlinearly interpolate between the 4 rays.

	      //Going with the latter:

	      //Now, with the radiation length, extract the two dose values on either side

	      //Check the borders - rvol should have an extra "border" for this purpose.
	      //If any rays are these added borders, it is outside dose and is background
	      if ( (ray_lookup[i][0]==0) || (ray_lookup[i][0]==ires[0]-1) ||
		   (ray_lookup[i][1]==0) || (ray_lookup[i][1]==ires[1]-1) )  {
		master_square[i/2][i%2] = background;
	      }

	      
	      else {

		dummy_lin_ex = ray_rad_len[i]-floor(ray_rad_len[i]);
		wijk[0] = ray_lookup[i][0] - 1;
		wijk[1] = ray_lookup[i][1] - 1;
		
		wijk[2] = (int) floor(ray_rad_len[i]);
		dummy_index1 = volume_index ( wed_vol->dim, wijk );
		wijk[2] = (int) ceil(ray_rad_len[i]);
		dummy_index2 = volume_index ( wed_vol->dim, wijk );

		master_square[i/2][i%2] = wed_vol_img[dummy_index1] * (1-dummy_lin_ex) + wed_vol_img[dummy_index2] * dummy_lin_ex;

		//		printf("wijk is %d %d %d at voxel %d\n",wijk[0],wijk[1],wijk[2], dummy_index2);

	      }
	      
	    }
	    
	    //Bilear interpolation from the square of wed dose ray values


	    dew_vol_img[didx] = (float)
	      ( master_square[0][0]*(1-master_coord[0])*(1-master_coord[1]) +
		master_square[0][1]*(master_coord[0])*(1-master_coord[1]) + 
		master_square[1][0]*(1-master_coord[0])*(master_coord[1]) + 
		master_square[1][1]*(master_coord[0])*(master_coord[1]) );

	    
	  }
	  //      printf("1 coord is %f %f %f at voxel %d\n",ap_coord[0],ap_coord[1],ap_coord[2], didx);
	  //	printf("coord is %f %f %f at voxel %d\n",ap_coord[0],ap_coord[1],ap_coord[2], didx);
	  //	printf("coord is %f %f %f at voxel %d\n",coord[0],coord[1],coord[2], didx);
	  //      printf("dijk is %d %d %d at voxel %d\n",dijk[0],dijk[1],dijk[2], didx);
	
      }
    }
  }
  
  

}

Volume* 
Rpl_volume::get_volume ()
{
    return d_ptr->proj_vol->get_volume ();
}

Proj_volume* 
Rpl_volume::get_proj_volume ()
{
    return d_ptr->proj_vol;
}

void
Rpl_volume::save (const char *filename)
{
    d_ptr->proj_vol->save (filename);
}

void
Rpl_volume::save (const std::string& filename)
{
    this->save (filename.c_str());
}

static float
lookup_attenuation_weq (float density)
{
    const double min_hu = -1000.0;
    if (density <= min_hu) {
        return 0.0;
    } else {
        return ((density + 1000.0)/1000.0);
    }
}

static float
lookup_attenuation (float density)
{
    return lookup_attenuation_weq (density);
}

static
void
rpl_ray_trace_callback (
    void *callback_data, 
    size_t vox_index, 
    double vox_len, 
    float vox_value
)
{
    Callback_data *cd = (Callback_data *) callback_data;
    Rpl_volume *rpl_vol = cd->rpl_vol;
    Ray_data *ray_data = cd->ray_data;
    float *depth_img = (float*) rpl_vol->get_volume()->img;
    int ap_idx = ray_data->ap_idx;
    int ap_area = cd->ires[0] * cd->ires[1];
    int step_num = vox_index + cd->step_offset;

    cd->accum += vox_len * lookup_attenuation (vox_value);

#if defined (commentout)
    if (ap_idx == 99 || ap_idx == 90) {
	printf ("%d %4d: %20g %20g\n", ap_idx, step_num, 
	    vox_value, cd->accum);
    }
#endif

#if defined (commentout)
    if (ap_idx >= 600) {
    printf ("--\ndim = %d %d %d\n", 
        rpl_vol->get_volume()->dim[0],
        rpl_vol->get_volume()->dim[1],
        rpl_vol->get_volume()->dim[2]);
    printf ("ap_area = %d, step_num = %d, ap_idx = %d\n", 
        ap_area, step_num, ap_idx);
    }
#endif

    /* GCS FIX: I have a rounding error somewhere -- maybe step_num
       starts at 1?  Or maybe proj_vol is not big enough?  
       This is a workaround until I can fix. */
    if (step_num >= rpl_vol->get_volume()->dim[2]) {
        return;
    }

    depth_img[ap_area*step_num + ap_idx] = cd->accum;
}

void
Rpl_volume::ray_trace (
    Volume *ct_vol,              /* I: CT volume */
    Ray_data *ray_data,          /* I: Pre-computed data for this ray */
    Volume_limit *vol_limit,     /* I: CT bounding region */
    const double *src,           /* I: @ source */
    int* ires                    /* I: ray cast resolution */
)
{
    Callback_data cd;

    if (!ray_data->intersects_volume) {
        return;
    }

    /* init callback data for this ray */
    cd.rpl_vol = this;
    cd.ray_data = ray_data;
    cd.accum = 0.0f;
    cd.ires = ires;

    /* Figure out how many steps to first step within volume */
    double dist = ray_data->front_dist - d_ptr->front_clipping_dist;
    cd.step_offset = (int) ceil (dist / d_ptr->proj_vol->get_step_length ());

#if VERBOSE
    printf ("front_dist = %f\n", ray_data->front_dist);
    printf ("front_clip = %f\n", d_ptr->front_clipping_dist);
    printf ("dist = %f\n", dist);
    printf ("step_offset = %d\n", cd.step_offset);
#endif
	
    /* Find location of first step within volume */
    double tmp[3];
    double first_loc[3];
    vec3_scale3 (tmp, ray_data->ray, 
        cd.step_offset * d_ptr->proj_vol->get_step_length ());
    vec3_add3 (first_loc, ray_data->p2, tmp);

#if VERBOSE
    printf ("first_loc = (%f, %f, %f)\n", 
        first_loc[0], first_loc[1], first_loc[2]);
#endif

    /* get radiographic depth along ray */
    ray_trace_uniform (
        ct_vol,                     // INPUT: CT volume
        vol_limit,                  // INPUT: CT volume bounding box
        &rpl_ray_trace_callback,    // INPUT: step action cbFunction
        &cd,                        // INPUT: callback data
        first_loc,                  // INPUT: ray starting point
        ray_data->ip2,              // INPUT: ray ending point
        d_ptr->proj_vol->get_step_length()); // INPUT: uniform ray step size
}



/* GCS FIX: farm these out to a separate file */
#if defined (commentout)
static float
lookup_attenuation_weq (float density)
{
    const double min_hu = -1000.0;
    if (density <= min_hu) {
        return 0.0;
    } else {
        return ((density + 1000.0)/1000.0);
    }
}

static float
lookup_attenuation (float density)
{
    return lookup_attenuation_weq (density);
}
#endif
