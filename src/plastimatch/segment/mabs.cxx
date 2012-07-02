/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsegment_config.h"
#include <list>
#include <stdio.h>
#include <stdlib.h>

#include "dir_list.h"
#include "file_util.h"
#include "itk_image_save.h"
#include "itk_threshold.h"
#include "mabs.h"
#include "mabs_parms.h"
#include "mabs_vote.h"
#include "plm_image.h"
#include "plm_parms.h"
#include "plm_stages.h"
#include "plm_warp.h"
#include "print_and_exit.h"
#include "registration_data.h"
#include "rtds.h"
#include "rtss.h"
#include "rtss_structure_set.h"
#include "string_util.h"
#include "xform.h"

Mabs::Mabs () { }

Mabs::~Mabs () { }

void
Mabs::run (const Mabs_parms& parms)
{
    /* Do a few sanity checks */
    if (!is_directory (parms.atlas_dir)) {
        print_and_exit ("Atlas dir (%s) is not a directory\n",
            parms.atlas_dir);
    }
    if (!file_exists (parms.registration_config)) {
        print_and_exit ("Couldn't find registration config (%s)\n", 
            parms.registration_config);
    }

    /* Make sure there is an output directory */
    Pstring out_dir = parms.labeling_output_fn;
    if (out_dir.empty()) {
        out_dir = "mabs";
    }

    /* Load the labeling file.  For now, we'll assume this is successful. */
    Plm_image fixed_image (parms.labeling_input_fn);

    /* Make a list to store voting results */
    std::vector<Mabs_vote*> vote_list;

    /* Loop through images in the atlas directory */
    Dir_list d (parms.atlas_dir);
    for (int i = 0; i < d.num_entries; i++) {
        Rtds rtds;

        /* Skip "." and ".." */
        if (!strcmp (d.entries[i], ".") || !strcmp (d.entries[i], "..")) {
            printf ("Skipping %s\n", d.entries[i]);
            continue;
        }

        /* Build string containing full path to atlas item */
        std::string path = compose_filename (parms.atlas_dir, d.entries[i]);

        /* Only consider directories */
        if (!is_directory (path.c_str())) {
            printf ("Skipping [2] %s\n", d.entries[i]);
            continue;
        }

        /* For now, only handle dicom directories.  We assume the 
           load is successful. */
        lprintf ("MABS loading %s\n", d.entries[i]);
        rtds.load_dicom_dir (path.c_str());

        /* Make a registration command string */
        std::string command_string = slurp_file (parms.registration_config);

        /* Parse the registration command string */
        Registration_parms regp;
        int rc = regp.set_command_string (command_string);
        if (rc) {
            print_and_exit ("Failure parsing command file: %s\n",
                parms.registration_config);
        }

        /* Manually set input files */
        Registration_data regd;
        regd.fixed_image = &fixed_image;
        regd.moving_image = rtds.m_img;

        /* Run the registration */
        Xform *xf_out;
        printf ("DO_REGISTRATION_PURE\n");
        printf ("regp.num_stages = %d\n", regp.num_stages);
        do_registration_pure (&xf_out, &regd, &regp);

        /* Warp the output image */
        printf ("Warp output image...\n");
        Plm_image_header fixed_pih (regd.fixed_image);
        Plm_image warped_image;
        plm_warp (&warped_image, 0, xf_out, &fixed_pih, regd.moving_image, 
            regp.default_value, 0, 1);

        /* Warp the structures */
        printf ("Warp structures...\n");
        Plm_image_header source_pih (rtds.m_img);
        rtds.m_rtss->prune_empty ();
        rtds.m_rtss->rasterize (&source_pih, false, false);
        rtds.m_rtss->warp (xf_out, &fixed_pih);

        /* Loop through structures for this atlas image */
        printf ("Vote...\n");
        for (size_t i = 0; i < rtds.m_rtss->get_num_structures(); i++) 
        {
            /* TODO: Check structure name - for now we just assume 
               structures are in the same order for each atlas */

            /* Make a new voter if needed */
            Mabs_vote *vote;
            if (i >= vote_list.size()) {
                vote = new Mabs_vote;
                vote_list.push_back (vote);
                vote->set_fixed_image (regd.fixed_image->itk_float());
            } else {
                vote = vote_list[i];
            }

            /* Extract structure as binary mask */
            UCharImageType::Pointer structure_image 
                = rtds.m_rtss->get_structure_image (i);

            /* Vote for each voxel */
            vote->vote (
                warped_image.itk_float(),
                structure_image);
        }

        /* Don't let regd destructor delete our fixed image */
        regd.fixed_image = 0;

        /* Clean up */
        delete xf_out;
    }

    /* Get output image for each label */
    lprintf ("Normalizing and saving weights\n");
    for (size_t i = 0; i < vote_list.size(); i++) {
        Mabs_vote *vote = vote_list[i];
        lprintf ("Normalizing votes\n");
        vote->normalize_votes();

        /* Optionally, get the weight image */
        FloatImageType::Pointer wi = vote->get_weight_image ();

        /* Optionally, save the weight files */
        if (parms.debug) {
            lprintf ("Saving weights\n");
            Pstring fn; 
            fn.format ("%s/weight_%04d.nrrd", out_dir.c_str(), i);
            itk_image_save (wi, fn.c_str());
        }

        /* Threshold the weight image */
        UCharImageType::Pointer thresh = itk_threshold_above (wi, 0.5);

        /* Optionally, save the thresholded files */
        /* GCS FIX: After we can create the structure set, we'll make 
           this optional */
        lprintf ("Saving thresholded structures\n");
        Pstring fn; 
        fn.format ("%s/label_%04d.nrrd", out_dir.c_str(), i);
        itk_image_save (thresh, fn.c_str());

        /* Assemble into structure set */
    }
}
