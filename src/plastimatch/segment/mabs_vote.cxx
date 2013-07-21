/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsegment_config.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif
#include "itkImageRegionIterator.h"

#include "logfile.h"
#include "mabs_subject.h"
#include "mabs_vote.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_math.h"
#include "volume.h"

class Plm_image;

class Mabs_vote_private {
public:
    Mabs_vote_private () {
        rho = 1;
        sigma = 50;
        minimum_similarity = 0.0001;
        like0_img = 0;
        like1_img = 0;
        score_img = 0;
    }
    ~Mabs_vote_private () {
        delete score_img;
        delete like0_img;
        delete like1_img;
    }
public:
    FloatImageType::Pointer target;
//    FloatImageType::Pointer like0;
//    FloatImageType::Pointer like1;
//    FloatImageType::Pointer weights;

    Plm_image *score_img;
    Plm_image *like0_img;
    Plm_image *like1_img;

    double rho;
    double sigma;
    double minimum_similarity;
};

Mabs_vote::Mabs_vote ()
{
    d_ptr = new Mabs_vote_private;
}

Mabs_vote::~Mabs_vote ()
{
    delete d_ptr;
}

void
Mabs_vote::set_fixed_image (
    FloatImageType::Pointer target
)
{
    /* Save a copy of target */
    d_ptr->target = target;

    /* Create a like0 image */
    d_ptr->like0_img = new Plm_image (
        PLM_IMG_TYPE_ITK_FLOAT, 
        Plm_image_header (d_ptr->target));

    /* Create a like1 image */
    d_ptr->like1_img = new Plm_image (
        PLM_IMG_TYPE_ITK_FLOAT, 
        Plm_image_header (d_ptr->target));
}

void
Mabs_vote::set_rho (
    float rho
)
{
    d_ptr->rho = (double) rho;
}

void
Mabs_vote::set_sigma (
    float sigma
)
{
    d_ptr->sigma = (double) sigma;
}

void
Mabs_vote::set_minimum_similarity (
    float minimum_similarity
)
{
    d_ptr->minimum_similarity = (double) minimum_similarity;
}

void
Mabs_vote::vote (
    FloatImageType::Pointer atlas_image, 
    FloatImageType::Pointer dmap_image
)
{
    /* GCS FIX: I shouldn't need to re-convert these every time I vote */
    Plm_image atl_i (atlas_image);
    Plm_image dmp_i (dmap_image);
    Plm_image tgt_i (d_ptr->target);

    Volume *atl_vol = atl_i.get_vol_float ();
    Volume *dmp_vol = dmp_i.get_vol_float ();
    Volume *tgt_vol = tgt_i.get_vol_float ();

    float *atl_img = (float*) atl_vol->img;
    float *dmp_img = (float*) dmp_vol->img;
    float *tgt_img = (float*) tgt_vol->img;

    Volume *like0_vol = d_ptr->like0_img->get_vol_float ();
    Volume *like1_vol = d_ptr->like1_img->get_vol_float ();

    float *like0_img = (float*) like0_vol->img;
    float *like1_img = (float*) like1_vol->img;
    
    plm_long v;
#pragma omp parallel for
    for (v = 0; v < tgt_vol->npix; v++) {
        /* Compute similarity between target and atlas images */
        double intensity_diff = tgt_img[v] - atl_img[v];

        /* GCS Note:  Sometimes value is zero, when med_diff is very high.
           When this happens for all atlas examples, we get divide by zero.
           Furthermore, there is no sense dividing by 
           M_SQRT2PI * d_ptr->sigma, because that is a constant. */
#if defined (commentout)
        similarity_value = exp (-(intensity_diff * intensity_diff) 
            / (2.0*d_ptr->sigma*d_ptr->sigma))
            / (M_SQRT2PI * d_ptr->sigma);
#endif

        /* So instead, we do this */
        double similarity_value = exp (-(intensity_diff * intensity_diff) 
            / (2.0*d_ptr->sigma*d_ptr->sigma));
        if (similarity_value < 0.0001) {
            similarity_value = 0.0001;
        }

        /* The above is too sensitive.  We should truncate. 
           Mixed Gaussian / Uniform model. */
        if (similarity_value < d_ptr->minimum_similarity) {
            similarity_value = d_ptr->minimum_similarity;
        }

        /* Compute the chance of being in the structure. */
        double dmap_value = dmp_img[v];
        /* GCS 2013-01-31: Distance map is squared.  This should 
           probably be corrected before saving */
        if (dmap_value > 0) {
            dmap_value = sqrt(dmap_value);
        } else {
            dmap_value = -sqrt(-dmap_value);
        }

        /* Nb. we need to check to make sure exp(dmap_value) 
           doesn't overflow.  The actual overflow is at about exp(700) 
           for double, and about exp(85) for float.  But we can be 
           a little more conservative. */
        double label_likelihood_0, label_likelihood_1;
        if (dmap_value > 50) {
            label_likelihood_0 = 0;
            label_likelihood_1 = 1;
        } else if (dmap_value > -50) {
            label_likelihood_0 = exp (-d_ptr->rho*dmap_value);
            label_likelihood_1 = exp (+d_ptr->rho*dmap_value);
        } else {
            label_likelihood_0 = 1;
            label_likelihood_1 = 0;
        }

        /* Compute total score, weighted by image similarity */
        double sum = label_likelihood_0 + label_likelihood_1;
        double l0 = (label_likelihood_0 / sum) * similarity_value;
        double l1 = (label_likelihood_1 / sum) * similarity_value;

        /* Save the scores */
        like0_img[v] += l0;
        like1_img[v] += l1;
    }
}

void
Mabs_vote::normalize_votes ()
{
    /* Create weight image */
    d_ptr->score_img = new Plm_image (
        PLM_IMG_TYPE_ITK_FLOAT, 
        Plm_image_header (d_ptr->target));

    Volume *score_vol = d_ptr->score_img->get_vol_float ();
    Volume *like0_vol = d_ptr->like0_img->get_vol_float ();
    Volume *like1_vol = d_ptr->like1_img->get_vol_float ();
    float *score_img = (float*) score_vol->img;
    float *like0_img = (float*) like0_vol->img;
    float *like1_img = (float*) like1_vol->img;
    
    plm_long v;
    float l0_min = FLT_MAX;
    float l1_min = FLT_MAX;
    float l0_max = -FLT_MAX;
    float l1_max = -FLT_MAX;
#pragma omp parallel
    {
        float priv_l0_min = FLT_MAX;
        float priv_l1_min = FLT_MAX;
        float priv_l0_max = -FLT_MAX;
        float priv_l1_max = -FLT_MAX;
#pragma omp for
        for (v = 0; v < score_vol->npix; v++) {
            float l0 = like0_img[v];
            float l1 = like1_img[v];
            score_img[v] = l1 / (l0 + l1);
            if (l0 < priv_l0_min) priv_l0_min = l0;
            if (l1 < priv_l1_min) priv_l1_min = l1;
            if (l0 > priv_l0_max) priv_l0_max = l0;
            if (l1 > priv_l1_max) priv_l1_max = l1;
        }

        /* GCS Note: this could be done faster using pragma omp flush,
           and then testing whether or not it is needed to enter the 
           critical section.  Cf: http://pc51.plm.eecs.uni-kassel.de/plm/fileadmin/pm/courses/pvI07/SL06.pdf
           However, for readability I leave it as-is
        */
#pragma omp critical
        { l0_min = std::min (l0_min, priv_l0_min); }
#pragma omp critical
        { l1_min = std::min (l1_min, priv_l1_min); }
#pragma omp critical
        { l0_max = std::max (l0_max, priv_l0_max); }
#pragma omp critical
        { l1_max = std::max (l1_max, priv_l1_max); }
    }
    lprintf ("\tLikelihood 0 \\in [ %g, %g ]\n", l0_min, l0_max);
    lprintf ("\tLikelihood 1 \\in [ %g, %g ]\n", l1_min, l1_max);
}

FloatImageType::Pointer
Mabs_vote::get_weight_image ()
{
    FloatImageType::Pointer itk_score_img = d_ptr->score_img->itk_float();
    return itk_score_img;
}
