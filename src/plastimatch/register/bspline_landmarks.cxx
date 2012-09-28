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

#include "file_util.h"
#include "plm_math.h"
#include "print_and_exit.h"
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

    FILE* landmark_fp = 0;
    static int it = 0;
    if (parms->debug) {
        char buf[1024];
        
        //sprintf (buf, "lm_%02d.txt", it);
        sprintf (buf, "%02d_lm_%02d.txt",
            parms->debug_stage, bst->feval);
        std::string fn = parms->debug_dir + "/" + buf;
        landmark_fp = plm_fopen (fn.c_str(), "wb");
        it ++;
    }

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

        if (parms->debug) {
            fprintf (landmark_fp, 
                "    flm  = %5.2f %5.2f %5.2f\n", 
                blm->fixed_landmarks->point_list[lidx].p[0],
                blm->fixed_landmarks->point_list[lidx].p[1],
                blm->fixed_landmarks->point_list[lidx].p[2]);
            fprintf (landmark_fp, 
                "    dxyz = %5.2f %5.2f %5.2f\n", 
                dxyz[0], dxyz[1], dxyz[2]);
            fprintf (landmark_fp, 
                "    diff = %5.2f %5.2f %5.2f (%5.2f)\n", 
                diff[0], diff[1], diff[2], sqrt(l_dist));
            fprintf (landmark_fp, 
                "    mxyz = %5.2f %5.2f %5.2f\n", 
                mxyz[0], mxyz[1], mxyz[2]);
            fprintf (landmark_fp, 
                "    mlm  = %5.2f %5.2f %5.2f\n", 
                blm->moving_landmarks->point_list[lidx].p[0],
                blm->moving_landmarks->point_list[lidx].p[1],
                blm->moving_landmarks->point_list[lidx].p[2]);
            fprintf (landmark_fp, "--\n");
        }

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

    if (parms->debug) {
        fclose (landmark_fp);
    }

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
