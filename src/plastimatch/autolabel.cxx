/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkImageRegionIterator.h"
#include "dlib/data_io.h"
#include "dlib/svm.h"

#include "autolabel.h"
#include "autolabel_ransac_est.h"
#include "bstring_util.h"
#include "dlib_trainer.h"
#include "itk_image.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "pstring.h"
#include "thumbnail.h"

/* ITK typedefs */
typedef itk::ImageRegionConstIterator< FloatImageType > FloatIteratorType;

/* Load the network into "dlib_network" (first argument) */
static void
load_dlib_network (
    dlib::decision_function< Dlib_trainer::Kernel_type > *dlib_network,
    const Pstring& network_fn)
{
    std::ifstream fin ((const char*) network_fn, std::ios::binary);
    deserialize (*dlib_network, fin);
}

static void
autolabel_tsv1 (Autolabel_parms *parms)
{
    FILE *fp;
    Pstring network_fn;

    /* Load network */
    network_fn.format ("%s/tsv1.net", parms->network_dir.c_str());
    dlib::decision_function< Dlib_trainer::Kernel_type > dlib_network;
    load_dlib_network (&dlib_network, network_fn);

    /* Load input image */
    Plm_image pli ((const char*) parms->input_fn, PLM_IMG_TYPE_ITK_FLOAT);

    Thumbnail thumbnail;
    thumbnail.set_input_image (&pli);
    thumbnail.set_thumbnail_dim (16);
    thumbnail.set_thumbnail_spacing (25.0f);

    /* Open output file (txt format) */
    fp = fopen ((const char*) parms->output_csv_fn, "w");
    if (!fp) {
        print_and_exit ("Failure to open file for write: %s\n", 
            (const char*) parms->output_csv_fn);
    }

    /* Create a vector to hold the results */
    Autolabel_point_vector apv;

    /* Loop through slices, and predict location for each slice */
    Plm_image_header pih (&pli);
    for (int i = 0; i < pih.Size(2); i++) {

        /* Create slice thumbnail */
        float loc = pih.m_origin[2] + i * pih.m_spacing[2];
        thumbnail.set_slice_loc (loc);
        FloatImageType::Pointer thumb_img = thumbnail.make_thumbnail ();

        /* Convert to dlib sample type */
        Dlib_trainer::Dense_sample_type d;
        FloatIteratorType it (thumb_img, thumb_img->GetLargestPossibleRegion());
        for (int j = 0; j < 256; j++) {
            d(j) = it.Get();
            ++it;
        }

        /* Predict the value */
        Autolabel_point ap;
        ap[0] = loc;
        ap[1] = dlib_network (d);
        ap[2] = 0.;
        apv.push_back (ap);
    }

    /* Run RANSAC to refine the estimate */
    if (parms->enforce_anatomic_constraints) {
        autolabel_ransac_est (apv);
    }

    /* Save the (debugging) output to a file */
    Autolabel_point_vector::iterator it;
    for (it = apv.begin(); it != apv.end(); it++) {
        fprintf (fp, "%g,%g,%g\n", (*it)[0], (*it)[1], (*it)[2]);
    }
    
    fclose (fp);
}

static void
autolabel_tsv2 (Autolabel_parms *parms)
{
    Labeled_pointset points;
    Pstring network_fn;

    /* Load x & y networks */
    network_fn.format ("%s/tsv2_x.net", parms->network_dir.c_str());
    dlib::decision_function< Dlib_trainer::Kernel_type > dlib_network_x;
    load_dlib_network (&dlib_network_x, network_fn);
    network_fn.format ("%s/tsv2_y.net", parms->network_dir.c_str());
    dlib::decision_function< Dlib_trainer::Kernel_type > dlib_network_y;
    load_dlib_network (&dlib_network_y, network_fn);

    /* Load input image */
    Plm_image pli ((const char*) parms->input_fn, PLM_IMG_TYPE_ITK_FLOAT);

    Thumbnail thumbnail;
    thumbnail.set_input_image (&pli);
    thumbnail.set_thumbnail_dim (16);
    thumbnail.set_thumbnail_spacing (25.0f);

    /* Loop through slices, and predict location for each slice */
    Plm_image_header pih (&pli);
    printf ("Looping...\n");
    for (int i = 0; i < pih.Size(2); i++) {

        /* Create slice thumbnail */
        float loc = pih.m_origin[2] + i * pih.m_spacing[2];
        thumbnail.set_slice_loc (loc);
        FloatImageType::Pointer thumb_img = thumbnail.make_thumbnail ();

        /* Convert to dlib sample type */
        Dlib_trainer::Dense_sample_type d;
        FloatIteratorType it (thumb_img, thumb_img->GetLargestPossibleRegion());
        for (int j = 0; j < 256; j++) {
            d(j) = it.Get();
            ++it;
        }

        /* Predict the value */
        Pstring label;
        label.format ("P_%02d", i);
        points.insert_lps (label.c_str(), 
            dlib_network_x (d), dlib_network_y (d), loc);
    }
    printf ("Done.\n");

    /* Save the pointset output to a file */
    if (parms->output_fcsv_fn.not_empty()) {
        points.save_fcsv (parms->output_fcsv_fn);
    }
}

void
autolabel (Autolabel_parms *parms)
{
    if (parms->task == "tsv1") {
        autolabel_tsv1 (parms);
    }
    else if (parms->task == "tsv2") {
        autolabel_tsv2 (parms);
    }
    else {
        printf ("Error, unknown autolabel task?\n");
    }
}
