/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "plmbase.h"
#include "plmregister.h"
#include "plmsys.h"

#include "plm_math.h"
#include "volume_macros.h"

void
bspline_landmarks_score_a (
    Bspline_parms *parms, 
    Bspline_state *bst, 
    Bspline_xform* bxf
)
{
    Bspline_score* ssd = &bst->ssd;
    Bspline_landmarks *blm = parms->blm;
    int lidx;
#if defined (commentout)
    FILE *fp, *fp2;
#endif
    float land_score, land_grad_coeff;

    land_score = 0;
    land_grad_coeff = blm->landmark_stiffness / blm->num_landmarks;

#if defined (commentout)
    fp  = fopen ("warplist_a.fcsv","w");
    fp2 = fopen ("distlist_a.dat","w");
    fprintf (fp, "# name = warped\n");
#endif

    for (lidx=0; lidx < blm->num_landmarks; lidx++) {
        plm_long p[3], q[3];
        plm_long qidx;
        float mxyz[3];   /* Location of fixed landmark in moving image */
        float diff[3];   /* mxyz - moving_landmark */
        float dc_dv[3];
        float dxyz[3];
        float l_dist=0;

        for (int d = 0; d < 3; d++) {
            p[d] = blm->fixed_landmarks_p[lidx*3+d];
            q[d] = blm->fixed_landmarks_q[lidx*3+d];
        }

        qidx = volume_index (bxf->vox_per_rgn, q);
        bspline_interp_pix (dxyz, bxf, p, qidx);

        for (int d = 0; d < 3; d++) {
            mxyz[d] = blm->fixed_landmarks->point_list[lidx].p[d] + dxyz[d];
            diff[d] = blm->moving_landmarks->point_list[lidx].p[d] - mxyz[d];
        }

        l_dist = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
        land_score += l_dist;

#if defined (commentout)
        printf ("\n");
        printf ("    flm  = %5.2f %5.2f %5.2f\n", 
            blm->fixed_landmarks->point_list[lidx].p[0],
            blm->fixed_landmarks->point_list[lidx].p[1],
            blm->fixed_landmarks->point_list[lidx].p[2]);
        printf ("    dxyz = %5.2f %5.2f %5.2f\n", 
            dxyz[0], dxyz[1], dxyz[2]);
        printf ("    diff = %5.2f %5.2f %5.2f (%5.2f)\n", 
            diff[0], diff[1], diff[2], sqrt(l_dist));
        printf ("    mxyz = %5.2f %5.2f %5.2f\n", 
            mxyz[0], mxyz[1], mxyz[2]);
        printf ("    mlm  = %5.2f %5.2f %5.2f\n", 
            blm->moving_landmarks->point_list[lidx].p[0],
            blm->moving_landmarks->point_list[lidx].p[1],
            blm->moving_landmarks->point_list[lidx].p[2]);
#endif

        // calculating gradients
        dc_dv[0] = - land_grad_coeff * diff[0];
        dc_dv[1] = - land_grad_coeff * diff[1];
        dc_dv[2] = - land_grad_coeff * diff[2];
        bspline_update_grad (bst, bxf, p, qidx, dc_dv);

#if defined (commentout)
        /* Note: Slicer landmarks are in RAS coordinates. Change LPS to RAS */
        fprintf (fp, "W%d,%f,%f,%f,1,1\n", lidx, -mxyz[0], -mxyz[1], mxyz[2]);
        fprintf (fp2,"W%d %.3f\n", lidx, sqrt(l_dist));
#endif
    }
#if defined (commentout)
    fclose(fp);
    fclose(fp2);
#endif

    ssd->lmetric = land_score / blm->num_landmarks;
}

void
bspline_landmarks_score (
    Bspline_parms *parms, 
    Bspline_state *bst, 
    Bspline_xform* bxf
)
{
    /* Only 'a' is supported at this time */
    bspline_landmarks_score_a (parms, bst, bxf);
}

void 
Bspline_landmarks::initialize (const Bspline_xform* bxf)
{
    if (!this->fixed_landmarks 
        || !this->moving_landmarks 
        || this->num_landmarks == 0)
    {
        return;
    }

    this->fixed_landmarks_p = new int[3*this->num_landmarks];
    this->fixed_landmarks_q = new int[3*this->num_landmarks];
    for (int i = 0; i < num_landmarks; i++) {
        for (int d = 0; d < 3; d++) {
            plm_long v;
            v = ROUND_INT ((this->fixed_landmarks->point_list[i].p[d] 
                    - bxf->img_origin[d]) / bxf->img_spacing[d]);
            printf ("(%f - %f) / %f = %u\n",
                this->fixed_landmarks->point_list[i].p[d], 
                bxf->img_origin[d], bxf->img_spacing[d], 
                (unsigned int) v);
            if (v < 0 || v >= bxf->img_dim[d])
            {
                print_and_exit (
                    "Error: fixed landmark %d outside of fixed image.\n", i);
            }
            this->fixed_landmarks_p[i*3+d] = v / bxf->vox_per_rgn[d];
            this->fixed_landmarks_q[i*3+d] = v % bxf->vox_per_rgn[d];
        }
    }
}

void 
Bspline_landmarks::set_landmarks (
    const Labeled_pointset *fixed_landmarks, 
    const Labeled_pointset *moving_landmarks)
{
    this->fixed_landmarks = fixed_landmarks;
    this->moving_landmarks = moving_landmarks;

    /* Find list with fewer landmarks */
    if (moving_landmarks->count() > fixed_landmarks->count()) {
        this->num_landmarks = fixed_landmarks->count();
    } else {
        this->num_landmarks = moving_landmarks->count();
    }
}

#if defined (commentout)
/*
  Moves moving landmarks according to the current vector field.
  Output goes into warped_landmarks and landvox_warp
  LW = warped landmark
  We must solve LW + u(LW) = LM to get new LW, corresponding to current vector field.
*/
void bspline_landmarks_warp (
    Volume *vector_field, 
    Bspline_parms *parms,
    Bspline_xform* bxf, 
    Volume *fixed, 
    Volume *moving)
{
    Bspline_landmarks *blm = parms->landmarks;
    int ri, rj, rk;
    int fi, fj, fk;
    int mi, mj, mk;
    float fx, fy, fz;
    float mx, my, mz;
    int i,d,fv, lidx;
    float dd, *vf, dxyz[3], *dd_min;

    if (vector_field->pix_type != PT_VF_FLOAT_INTERLEAVED)
        print_and_exit ("Sorry, this type of vector field is not supported in landmarks_warp\n");	
    vf = (float *)vector_field->img;

    dd_min = (float *)malloc( blm->num_landmarks * sizeof(float));
    for (d=0;d<blm->num_landmarks;d++) dd_min[d] = 1e20F; //a very large number

    for (rk = 0, fk = bxf->roi_offset[2]; rk < bxf->roi_dim[2]; rk++, fk++) {
        fz = bxf->img_origin[2] + bxf->img_spacing[2] * fk;
        for (rj = 0, fj = bxf->roi_offset[1]; rj < bxf->roi_dim[1]; rj++, fj++) {
            fy = bxf->img_origin[1] + bxf->img_spacing[1] * fj;
            for (ri = 0, fi = bxf->roi_offset[0]; ri < bxf->roi_dim[0]; ri++, fi++) {
                fx = bxf->img_origin[0] + bxf->img_spacing[0] * fi;

                fv = fk * vector_field->dim[0] * vector_field->dim[1] 
                    + fj * vector_field->dim[0] +fi ;

                for (d=0;d<3;d++) dxyz[d] = vf[3*fv+d];

                /* Find correspondence in moving image */
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
                for (lidx = 0; lidx < blm->num_landmarks; lidx++) {
                    dd = (mi - blm->landvox_mov[lidx*3+0]) * (mi - blm->landvox_mov[lidx*3+0])
                        +(mj - blm->landvox_mov[lidx*3+1]) * (mj - blm->landvox_mov[lidx*3+1])
                        +(mk - blm->landvox_mov[lidx*3+2]) * (mk - blm->landvox_mov[lidx*3+2]);
                    if (dd < dd_min[lidx]) { 
                        dd_min[lidx]=dd;   
                        for (d=0;d<3;d++) {
                            blm->landmark_dxyz[3*lidx+d] = vf[3*fv+d];
                        }
                    }
                } 
            }
        }
    }

    for (i=0;i<blm->num_landmarks;i++)  {
        for (d=0; d<3; d++) {
            blm->warped_landmarks[3*i+d] 
                = blm->moving_landmarks->points[3*i+d]
                - blm->landmark_dxyz[3*i+d];
        }
    }

    /* calculate voxel positions of warped landmarks  */
    for (lidx = 0; lidx < blm->num_landmarks; lidx++) {
        for (d = 0; d < 3; d++) {
            blm->landvox_warp[lidx*3 + d] 
                = ROUND_INT ((blm->warped_landmarks[lidx*3 + d] 
                        - fixed->offset[d]) / fixed->spacing[d]);
            pp	    if (blm->landvox_warp[lidx*3 + d] < 0 
                || blm->landvox_warp[lidx*3 + d] >= fixed->dim[d])
            {
                print_and_exit (
                    "Error, warped landmark %d outside of fixed image for dim %d.\n"
                    "Location in vox = %d\n"
                    "Image boundary in vox = (%d %d)\n",
                    lidx, d, blm->landvox_warp[lidx*3 + d], 0, fixed->dim[d]-1);
            }
        } 
    }
    free (dd_min);
}
#endif
