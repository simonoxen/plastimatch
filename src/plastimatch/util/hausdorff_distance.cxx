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
#include "itk_image_load.h"
#include "itk_resample.h"
#include "logfile.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "volume.h"

class Hausdorff_distance_private {
public:
    Hausdorff_distance_private () {
        hausdorff_distance = 0.f;
        avg_hausdorff_distance = 0.f;
        pct_hausdorff_distance = 0.f;
        pct_hausdorff_distance_fraction = 0.95;
    }
public:
    float hausdorff_distance;
    float avg_hausdorff_distance;
    float pct_hausdorff_distance;
    float pct_hausdorff_distance_fraction;

    UCharImageType::Pointer ref_image;
    UCharImageType::Pointer cmp_image;
    
    FloatImageType::Pointer fwd_dmap;
    FloatImageType::Pointer rev_dmap;
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
    UCharImageType::Pointer image,
    FloatImageType::Pointer dmap
)
{
    /* Convert to Plm_image type */
    Plm_image pli_uchar (image);
    Volume *vol_uchar = pli_uchar.gpuit_uchar ();
    unsigned char *img_uchar = (unsigned char*) vol_uchar->img;
    Plm_image pli_dmap (dmap);
    Volume *vol_dmap = pli_dmap.gpuit_float ();
    float *img_dmap = (float*) vol_dmap->img;

    /* Make an array to store the distances */
    float *distance_array = new float[vol_uchar->npix];

    /* Loop through voxels, find distances */
    float max_distance = 0;
    double sum_distance = 0;
    plm_long num_vox = 0;
    for (plm_long i = 0; i < vol_uchar->npix; i++) {
        if (img_uchar[i]) {
            float dist = 0;
            if (img_dmap[i] > 0) {
                dist = img_dmap[i];
            }
            if (img_dmap[i] > max_distance) {
                max_distance = img_dmap[i];
            }
            sum_distance += dist;
            distance_array[num_vox] = dist;
            num_vox ++;
        }
    }

    /* Figure out HD95 stuff */
    float hd_pct = 0;
    if (num_vox > 0) {
        int ordinal = (int) floor (
            d_ptr->pct_hausdorff_distance_fraction * num_vox-1);
        if (ordinal > num_vox - 1) {
            ordinal = num_vox - 1;
        }
        hd_pct = kth_smallest (distance_array, num_vox, ordinal);
    }

    /* Record results */
    if (max_distance > d_ptr->hausdorff_distance) {
        d_ptr->hausdorff_distance = max_distance;
    }
    if (num_vox > 0) {
        d_ptr->avg_hausdorff_distance += 0.5 * (sum_distance / num_vox);
        d_ptr->pct_hausdorff_distance += 0.5 * hd_pct;
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

    Distance_map dmap;
    dmap.set_input_image (d_ptr->cmp_image);
    dmap.set_inside_is_positive (false);
    dmap.set_use_squared_distance (false);
    dmap.run ();
    d_ptr->fwd_dmap = dmap.get_output_image ();

    dmap.set_input_image (d_ptr->ref_image);
    dmap.set_inside_is_positive (false);
    dmap.set_use_squared_distance (false);
    dmap.run ();
    d_ptr->rev_dmap = dmap.get_output_image ();

    d_ptr->hausdorff_distance = 0;
    d_ptr->avg_hausdorff_distance = 0;
    d_ptr->pct_hausdorff_distance = 0;
    this->run_internal (d_ptr->ref_image, d_ptr->fwd_dmap);
    this->run_internal (d_ptr->cmp_image, d_ptr->rev_dmap);
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

void 
Hausdorff_distance::debug ()
{
    lprintf (
	"Hausdorff distance = %f\n"
	"Average Hausdorff distance = %f\n"
	"Percent Hausdorff distance = %f\n",
	this->get_hausdorff (),
	this->get_average_hausdorff (),
	this->get_percent_hausdorff ());
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
