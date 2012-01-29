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
#include "print_and_exit.h"
#include "pstring.h"
#include "thumbnail.h"

/* ITK typedefs */
typedef itk::ImageRegionConstIterator< FloatImageType > FloatIteratorType;

void
autolabel_tsv1 (Autolabel_parms *parms)
{
    FILE *fp;

    /* Load network */
    //dlib::decision_function<kernel_type> dlib_network;
    dlib::decision_function< Dlib_trainer::Kernel_type > dlib_network;
    std::ifstream fin ((const char*) parms->network_fn, std::ios::binary);
    printf ("Trying to deserialize...\n");
    deserialize (dlib_network, fin);
    printf ("Done.\n");

    /* Load input image */
    Plm_image pli ((const char*) parms->input_fn, PLM_IMG_TYPE_ITK_FLOAT);

    Thumbnail thumbnail;
    thumbnail.set_input_image (&pli);
    thumbnail.set_thumbnail_dim (16);
    thumbnail.set_thumbnail_spacing (25.0f);

    /* Open output file (txt format) */
    fp = fopen ((const char*) parms->output_fn, "w");
    if (!fp) {
	print_and_exit ("Failure to open file for write: %s\n", 
	    (const char*) parms->output_fn);
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

    /* Save the output to a file */
    Autolabel_point_vector::iterator it;
    for (it = apv.begin(); it != apv.end(); it++) {
	fprintf (fp, "%g %g %g\n", (*it)[0], (*it)[1], (*it)[2]);
    }
    
    fclose (fp);
}

void
autolabel (Autolabel_parms *parms)
{
    if (parms->task == "tsv1") {
        autolabel_tsv1 (parms);
    }
    else if (parms->task == "tsv2") {
        autolabel_tsv1 (parms);
    }
    else {
        printf ("Error, unknown autolabel task?\n");
    }
}
