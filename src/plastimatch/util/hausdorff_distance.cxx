/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "itkImage.h"
#include "plm_HausdorffDistanceImageFilter.h"
#include "wirth.h"

#include "distance_map.h"
#include "hausdorff_distance.h"
#include "image_boundary.h"
#include "itk_image_load.h"
#include "itk_resample.h"
#include "logfile.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "volume.h"

class Hausdorff_distance_private {
public:
    Hausdorff_distance_private () {
        pct_hausdorff_distance_fraction = 0.95;
        this->clear_statistics ();
    }
public:
    void clear_statistics () {
        hausdorff_distance = 0.f;
        avg_hausdorff_distance = 0.f;
        pct_hausdorff_distance = 0.f;
        boundary_hausdorff_distance = 0.f;
        avg_boundary_hausdorff_distance = 0.f;
        pct_boundary_hausdorff_distance = 0.f;
    }
public:
    float hausdorff_distance;
    float avg_hausdorff_distance;
    float pct_hausdorff_distance;
    float boundary_hausdorff_distance;
    float avg_boundary_hausdorff_distance;
    float pct_boundary_hausdorff_distance;
    float pct_hausdorff_distance_fraction;

    UCharImageType::Pointer ref_image;
    UCharImageType::Pointer cmp_image;
};

Hausdorff_distance::Hausdorff_distance ()
{
    d_ptr = new Hausdorff_distance_private;
}

Hausdorff_distance::~Hausdorff_distance ()
{
    delete d_ptr;
}

void 
Hausdorff_distance::set_reference_image (const char* image_fn)
{
    d_ptr->ref_image = itk_image_load_uchar (image_fn, 0);
}

void 
Hausdorff_distance::set_reference_image (
    const UCharImageType::Pointer image)
{
    d_ptr->ref_image = image;
}

void 
Hausdorff_distance::set_compare_image (const char* image_fn)
{
    d_ptr->cmp_image = itk_image_load_uchar (image_fn, 0);
}

void 
Hausdorff_distance::set_compare_image (
    const UCharImageType::Pointer image)
{
    d_ptr->cmp_image = image;
}

void 
Hausdorff_distance::set_hausdorff_distance_fraction (
    float hausdorff_distance_fraction)
{
    d_ptr->pct_hausdorff_distance_fraction = hausdorff_distance_fraction;
}

void 
Hausdorff_distance::run_internal (
    UCharImageType::Pointer image1,
    UCharImageType::Pointer image2
)
{
    /* Compute distance map */
    Distance_map dmap_filter;
    dmap_filter.set_input_image (image2);
    dmap_filter.set_inside_is_positive (false);
    dmap_filter.set_use_squared_distance (false);
    dmap_filter.run ();
    FloatImageType::Pointer dmap = dmap_filter.get_output_image ();

    /* Convert to Plm_image type */
    Plm_image pli_uchar (image1);
    Volume *vol_uchar = pli_uchar.get_vol_uchar ();
    unsigned char *img_uchar = (unsigned char*) vol_uchar->img;
    Plm_image pli_dmap (dmap);
    Volume *vol_dmap = pli_dmap.get_vol_float ();
    float *img_dmap = (float*) vol_dmap->img;

    /* Find boundary pixels */
    Image_boundary ib;
    ib.set_input_image (image1);
    ib.run ();
    UCharImageType::Pointer itk_ib = ib.get_output_image ();

    /* Convert to plm_image */
    Plm_image pli_ib (itk_ib);
    Volume *vol_ib = pli_ib.get_vol_uchar ();
    unsigned char *img_ib = (unsigned char*) vol_ib->img;

    /* Make an array to store the distances */
    float *h_distance_array = new float[vol_uchar->npix];
    float *bh_distance_array = new float[vol_uchar->npix];

    /* Loop through voxels, find distances */
    float max_h_distance = 0;
    float max_bh_distance = 0;
    double sum_h_distance = 0;
    double sum_bh_distance = 0;
    plm_long num_h_vox = 0;
    plm_long num_bh_vox = 0;
    for (plm_long i = 0; i < vol_uchar->npix; i++) {
        if (!img_uchar[i]) {
            continue;
        }

        /* Get distance map value for this voxel */
        float h_dist = img_dmap[i];   /* dist for set hausdorff */
        float bh_dist = img_dmap[i];  /* dist for boundary hausdorff */
        if (img_dmap[i] < 0) {
            h_dist = 0;
            bh_dist = - bh_dist;
        }

        /* Update statistics for hausdorff */
        if (h_dist > max_h_distance) {
            max_h_distance = h_dist;
        }
        sum_h_distance += h_dist;
        h_distance_array[num_h_vox] = h_dist;
        num_h_vox ++;
        
        /* Update statistics for boundary hausdorff */
        if (img_ib[i]) {
            if (bh_dist > max_bh_distance) {
                max_bh_distance = bh_dist;
            }
            sum_bh_distance += bh_dist;
            bh_distance_array[num_bh_vox] = bh_dist;
            num_bh_vox ++;
        }
    }

    /* Figure out HD95 stuff */
    float h_pct = 0, bh_pct = 0;
    if (num_h_vox > 0) {
        int ordinal = (int) floor (
            d_ptr->pct_hausdorff_distance_fraction * num_h_vox-1);
        if (ordinal > num_h_vox - 1) {
            ordinal = num_h_vox - 1;
        }
        h_pct = kth_smallest (h_distance_array, num_h_vox, ordinal);
    }
    if (num_bh_vox > 0) {
        int ordinal = (int) floor (
            d_ptr->pct_hausdorff_distance_fraction * num_bh_vox-1);
        if (ordinal > num_bh_vox - 1) {
            ordinal = num_bh_vox - 1;
        }
        bh_pct = kth_smallest (bh_distance_array, num_bh_vox, ordinal);
    }

    /* Record results */
    if (max_h_distance > d_ptr->hausdorff_distance) {
        d_ptr->hausdorff_distance = max_h_distance;
    }
    if (max_bh_distance > d_ptr->boundary_hausdorff_distance) {
        d_ptr->boundary_hausdorff_distance = max_bh_distance;
    }
    if (num_h_vox > 0) {
        d_ptr->avg_hausdorff_distance += 0.5 * (sum_h_distance / num_h_vox);
        d_ptr->pct_hausdorff_distance += 0.5 * h_pct;
    }
    if (num_bh_vox > 0) {
        d_ptr->avg_boundary_hausdorff_distance 
            += 0.5 * (sum_bh_distance / num_bh_vox);
        d_ptr->pct_boundary_hausdorff_distance += 0.5 * bh_pct;
    }
}

void 
Hausdorff_distance::run ()
{
    /* Resample cmp image onto geometry of reference */
    if (!itk_image_header_compare (d_ptr->ref_image, d_ptr->cmp_image)) {
        d_ptr->cmp_image = resample_image (d_ptr->cmp_image, 
            Plm_image_header (d_ptr->ref_image), 0, 0);
    }

    d_ptr->clear_statistics ();
    this->run_internal (d_ptr->ref_image, d_ptr->cmp_image);
    this->run_internal (d_ptr->cmp_image, d_ptr->ref_image);
}

void 
Hausdorff_distance::run_obsolete ()
{
    typedef unsigned char T;
    typedef itk::plm_HausdorffDistanceImageFilter< 
	itk::Image<T,3>, itk::Image<T,3> > Hausdorff_filter;
    Hausdorff_filter::Pointer h_filter = Hausdorff_filter::New ();

    h_filter->SetInput1 (d_ptr->ref_image);
    h_filter->SetInput2 (d_ptr->cmp_image);
    h_filter->SetUseImageSpacing (true);
    try {
        h_filter->Update ();
    } catch (itk::ExceptionObject &err) {
	std::cout << "ITK Exception: " << err << std::endl;
        return;
    }
    d_ptr->hausdorff_distance 
        = h_filter->GetHausdorffDistance ();
    d_ptr->avg_hausdorff_distance 
        = h_filter->GetAverageHausdorffDistance ();
}

float 
Hausdorff_distance::get_hausdorff ()
{
    return d_ptr->hausdorff_distance;
}

float 
Hausdorff_distance::get_average_hausdorff ()
{
    return d_ptr->avg_hausdorff_distance;
}

float 
Hausdorff_distance::get_percent_hausdorff ()
{
    return d_ptr->pct_hausdorff_distance;
}

float 
Hausdorff_distance::get_boundary_hausdorff ()
{
    return d_ptr->boundary_hausdorff_distance;
}

float 
Hausdorff_distance::get_average_boundary_hausdorff ()
{
    return d_ptr->avg_boundary_hausdorff_distance;
}

float 
Hausdorff_distance::get_percent_boundary_hausdorff ()
{
    return d_ptr->pct_boundary_hausdorff_distance;
}

void 
Hausdorff_distance::debug ()
{
    lprintf (
	"Hausdorff distance = %f\n"
	"Average Hausdorff distance = %f\n"
	"Percent (%.2f) Hausdorff distance = %f\n"
	"Hausdorff distance (boundary) = %f\n"
	"Average Hausdorff distance (boundary) = %f\n"
	"Percent (%.2f) Hausdorff distance (boundary) = %f\n",
	this->get_hausdorff (),
	this->get_average_hausdorff (),
        d_ptr->pct_hausdorff_distance_fraction,
	this->get_percent_hausdorff (),
	this->get_boundary_hausdorff (),
	this->get_average_boundary_hausdorff (),
        d_ptr->pct_hausdorff_distance_fraction,
	this->get_percent_boundary_hausdorff ()
    );
}

void 
do_hausdorff (
    UCharImageType::Pointer image_1, 
    UCharImageType::Pointer image_2
)
{
    Hausdorff_distance hd;
    hd.set_reference_image (image_1);
    hd.set_compare_image (image_2);
    hd.run ();
    hd.debug ();
}
