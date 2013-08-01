/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <float.h>
#include <math.h>
#include "itkImageRegionIterator.h"

#include "itk_image.h"
#include "itk_image_type.h"
#include "pointset_warp.h"

void
pointset_warp (
    Labeled_pointset *warped_pointset,
    Labeled_pointset *input_pointset,
    DeformationFieldType::Pointer vf)
{
    float *dist_array = new float[input_pointset->count()];

    for (size_t i = 0; i < input_pointset->count(); i++) {
        /* Clone pointset (to set labels) */
        warped_pointset->insert_lps (
            input_pointset->point_list[i].get_label(),
            input_pointset->point_list[i].p[0],
            input_pointset->point_list[i].p[1],
            input_pointset->point_list[i].p[2]);

        /* Initialize distance array */
        dist_array[i] = FLT_MAX;
    }

    /* Loop through vector field */
    typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
    FieldIterator fi (vf, vf->GetLargestPossibleRegion());
    for (fi.GoToBegin(); !fi.IsAtEnd(); ++fi) {
        
        /* Compute location of correspondence in moving image */
	DeformationFieldType::IndexType idx = fi.GetIndex ();
	FloatPoint3DType fixed_location;
        FloatPoint3DType moving_location;
        vf->TransformIndexToPhysicalPoint (idx, fixed_location);
	const FloatVector3DType& dxyz = fi.Get();
        moving_location[0] = fixed_location[0] + dxyz[0];
        moving_location[1] = fixed_location[1] + dxyz[1];
        moving_location[2] = fixed_location[2] + dxyz[2];

        /* Loop through landmarks */
        for (size_t i = 0; i < input_pointset->count(); i++) {
            /* Get distance from correspondence to landmark */
            float dv[3] = {
                moving_location[0] - input_pointset->point_list[i].p[0],
                moving_location[1] - input_pointset->point_list[i].p[1],
                moving_location[2] - input_pointset->point_list[i].p[2]
            };
            float dist = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];

            /* Update correspondence if current voxel is closest so far */
            if (dist < dist_array[i]) {
                printf ("%d: %f -> %f\n",
                    (int) i, dist_array[i], dist);
                dist_array[i] = dist;
                warped_pointset->point_list[i].p[0] = fixed_location[0];
                warped_pointset->point_list[i].p[1] = fixed_location[1];
                warped_pointset->point_list[i].p[2] = fixed_location[2];
            }
        }
    }

    /* Loop through landmarks, refining estimate */
    for (size_t i = 0; i < input_pointset->count(); i++) {
        for (int its = 0; its < 10; its++) {
            
        }
    }

    delete[] dist_array;
}

#if defined (commentout)
void
calculate_warped_landmarks_by_vf (Landmark_warp *lw , Volume *vector_field)

// same as calculate_warped_landmarks, but accepts vector_field directly,
// not through Xform lw->m_vf

/*Moves moving landmarks according to the current vector field.
  Output goes into lw->m_warped_landmarks
  LW = warped landmark
  We must solve LW + u(LW) = LM to get new LW, corresponding to current vector field.

  Adapted from bspline_landmarks_warp(...) in bspline_landmarks.c to use Landmark_warp */

{
    plm_long ri, rj, rk;
    plm_long fi, fj, fk, fv;
    plm_long mi, mj, mk;
    float fx, fy, fz;
    float mx, my, mz;
    int i, d, lidx;
    float dd, *vf, dxyz[3], *dd_min;
    
    int num_landmarks;
    int *landvox_mov, *landvox_fix, *landvox_warp;
    float *warped_landmarks;
    float *landmark_dxyz;
//    Volume *vector_field;
    Volume *moving;
    plm_long fixed_dim[3];
    float fixed_spacing[3], fixed_offset[3], fixed_direction_cosines[9];

    num_landmarks = lw->m_fixed_landmarks->num_points;

    landvox_mov  = (int *)malloc( 3*num_landmarks * sizeof(int));
    landvox_fix  = (int *)malloc( 3*num_landmarks * sizeof(int));
    landvox_warp = (int *)malloc( 3*num_landmarks * sizeof(int));
    landmark_dxyz = (float *)malloc( 3*num_landmarks * sizeof(float));
    warped_landmarks = (float *)malloc( 3*num_landmarks * sizeof(float));

    moving = lw->m_input_img->get_vol_float ();

    // fixed dimensions set come from lw->m_pih //
    lw->m_pih.get_dim (fixed_dim);
    lw->m_pih.get_spacing (fixed_spacing);
    lw->m_pih.get_origin (fixed_offset);
    lw->m_pih.get_direction_cosines (fixed_direction_cosines);

    if (vector_field->pix_type != PT_VF_FLOAT_INTERLEAVED)
	print_and_exit ("Sorry, this type of vector field is not supported in landmarks_warp\n");	
    vf = (float *)vector_field->img;

    // fill in landvox'es //
    landmark_convert_mm_to_voxel (landvox_fix, lw->m_fixed_landmarks, 
	fixed_offset, fixed_spacing, fixed_dim, fixed_direction_cosines);
    landmark_convert_mm_to_voxel (landvox_mov, lw->m_moving_landmarks, 
	moving->offset, moving->spacing, moving->dim, 
	moving->direction_cosines);
    
    printf("done landvox; n=%d\n", num_landmarks);

    printf("fix offs %f %f %f\n",  fixed_offset[0],fixed_offset[1],fixed_offset[2]);
    printf("fix dim  %d %d %d\n",  
        (int) fixed_dim[0], (int) fixed_dim[1], (int) fixed_dim[2]);
    printf("mov offs %f %f %f\n", moving->offset[0], 
        moving->offset[1], moving->offset[2]);
    printf("vf dim  %d %d %d\n", (int) vector_field->dim[0], 
        (int) vector_field->dim[1], (int) vector_field->dim[2]);

    for(i=0;i<num_landmarks;i++) {
        printf("%d %d %d    %d %d %d\n",
            landvox_fix[3*i+0],landvox_fix[3*i+1],landvox_fix[3*i+2],
            landvox_mov[3*i+0],landvox_mov[3*i+1],landvox_mov[3*i+2]
        );
    }

    dd_min = (float *)malloc( num_landmarks * sizeof(float));
    for (d=0;d<num_landmarks;d++) dd_min[d] = 1e20F; //a very large number

    // roi_offset and roi_dim are not used here //

    for (rk = 0, fk = 0; rk < fixed_dim[2]; rk++, fk++) {
	fz = fixed_offset[2] + fixed_spacing[2] * fk;
	for (rj = 0, fj = 0; rj < fixed_dim[1]; rj++, fj++) {
	    fy = fixed_offset[1] + fixed_spacing[1] * fj;
	    for (ri = 0, fi = 0; ri < fixed_dim[0]; ri++, fi++) {
		fx = fixed_offset[0] + fixed_spacing[0] * fi;

		fv = fk * vector_field->dim[0] * vector_field->dim[1] 
		    + fj * vector_field->dim[0] +fi ;

		for (d=0;d<3;d++) dxyz[d] = vf[3*fv+d];

		// Find correspondence in moving image //
		mx = fx + dxyz[0];
		mi = ROUND_INT ((mx - moving->offset[0]) / moving->spacing[0]);
		if (mi < 0 || mi >= moving->dim[0]) continue;
		my = fy + dxyz[1];
		mj = ROUND_INT ((my - moving->offset[1]) / moving->spacing[1]);
		if (mj < 0 || mj >= moving->dim[1]) continue;
		mz = fz + dxyz[2];
		mk = ROUND_INT ((mz - moving->offset[2]) / moving->spacing[2]);
		if (mk < 0 || mk >= moving->dim[2]) continue;

		//saving vector field in a voxel which is the closest to landvox_mov
		//after being displaced by the vector field
		for (lidx = 0; lidx < num_landmarks; lidx++) {
		    dd = (mi - landvox_mov[lidx*3+0]) * (mi - landvox_mov[lidx*3+0])
			+(mj - landvox_mov[lidx*3+1]) * (mj - landvox_mov[lidx*3+1])
			+(mk - landvox_mov[lidx*3+2]) * (mk - landvox_mov[lidx*3+2]);
		    if (dd < dd_min[lidx]) { 
			dd_min[lidx]=dd;   
			for (d=0;d<3;d++) {
			    landmark_dxyz[3*lidx+d] = vf[3*fv+d];
			}
		    }
		} 
	    }
	}
    }

    for (i = 0; i < num_landmarks; i++) {
	for (d=0; d<3; d++) {
	    warped_landmarks[3*i+d]
		= lw->m_moving_landmarks->points[3*i+d]
		- landmark_dxyz[3*i+d];
	}
    }

    // calculate voxel positions of warped landmarks  //
    for (lidx = 0; lidx < num_landmarks; lidx++) {
	for (d = 0; d < 3; d++) {
	    landvox_warp[lidx*3 + d] 
		= ROUND_INT ((warped_landmarks[lidx*3 + d] 
			- fixed_offset[d]) / fixed_spacing[d]);
	    if (landvox_warp[lidx*3 + d] < 0 
		|| landvox_warp[lidx*3 + d] >= fixed_dim[d])
	    {
		print_and_exit (
		    "Error, warped landmark %d outside of fixed image for dim %d.\n"
		    "Location in vox = %d\n"
		    "Image boundary in vox = (%d %d)\n",
		    lidx, d, landvox_warp[lidx*3 + d], 0, fixed_dim[d]-1);
	    }
	} 
	pointset_add_point_noadjust (lw->m_warped_landmarks, warped_landmarks+3*lidx);
    }

    printf("done warping, printing rms\n");

//debug only
    fy = 0;
    for (lidx = 0; lidx < num_landmarks; lidx++)
    {
	fx=0;
	for (d = 0; d < 3; d++) { 
	    fz = (lw->m_fixed_landmarks->points[3*lidx+d] - lw->m_warped_landmarks->points[3*lidx+d] );
	    fx += fz*fz;
	}
	printf("landmark %3d err %f mm\n", lidx, sqrt(fx));
	fy+=fx;
    }
    printf("landmark RMS err %f mm\n", sqrt(fy/num_landmarks));
// end debug only

    free (dd_min);
    free (landvox_mov);
    free (landvox_warp);
    free (landvox_fix);
    free (landmark_dxyz);
    free (warped_landmarks);
}
#endif
