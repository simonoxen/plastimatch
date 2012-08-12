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

class Mabs_private {
public:
    std::map<std::string, Mabs_vote*> vote_map;
};

Mabs::Mabs () {
    d_ptr = new Mabs_private;
}

Mabs::~Mabs () {
    delete d_ptr;
}

/* Map an input-specific structure name to a canonical structure name 
   Return "" if no canonical name */
std::string
Mabs::map_structure_name (
    const Mabs_parms& parms, 
    const std::string& ori_name)
{
    if (parms.structure_map.size() == 0) {
        lprintf ("$ No structure list specified\n");
        return ori_name;
    }

    std::map<std::string, std::string>::const_iterator it 
        = parms.structure_map.find (ori_name);
    if (it == parms.structure_map.end()) {
        lprintf (" $ irrelevant structure: %s\n", ori_name.c_str());
        return "";
    }

    const std::string& mapped_name = it->second;
    if (mapped_name == "") {
        lprintf (" $ irrelevant structure: %s\n", ori_name.c_str());
    }
    else if (mapped_name == ori_name) {
        lprintf (" $ relevant structure: %s\n", ori_name.c_str());
    }
    else {
        lprintf (" $ relevant structure: %s -> %s\n", 
            ori_name.c_str(), mapped_name.c_str());
    }
    return mapped_name;
}

void
Mabs::run (const Mabs_parms& parms)
{
    /* Do a few sanity checks */
    if (!is_directory (parms.atlas_dir)) {
        print_and_exit ("Atlas dir (%s) is not a directory\n",
            parms.atlas_dir.c_str());
    }
    if (!file_exists (parms.registration_config)) {
        print_and_exit ("Couldn't find registration config (%s)\n", 
            parms.registration_config.c_str());
    }

    /* Make sure there is an output directory */
    std::string out_dir = parms.labeling_output_fn;
    if (out_dir == "") {
        out_dir = "mabs";
    }

    /* Load the labeling file.  For now, we'll assume this is successful. */
    Plm_image fixed_image (parms.labeling_input_fn);

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
        rtds.m_rtss->prune_empty ();

        /* Inspect the structures -- we might be able to skip the 
           atlas if it has no relevant structures */
        bool can_skip = true;
        for (size_t i = 0; i < rtds.m_rtss->get_num_structures(); i++) {
            std::string ori_name = rtds.m_rtss->get_structure_name (i);
            std::string mapped_name = this->map_structure_name (parms, 
                ori_name);
            if (mapped_name != "") {
                can_skip = false;
                break;
            }
        }
        if (can_skip) {
            lprintf ("No relevant structures. Skipping.\n");
            continue;
        }

        /* Make a registration command string */
        std::string command_string = slurp_file (parms.registration_config);

        /* Parse the registration command string */
        Registration_parms regp;
        int rc = regp.set_command_string (command_string);
        if (rc) {
            print_and_exit ("Failure parsing command file: %s\n",
                parms.registration_config.c_str());
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
        rtds.m_rtss->rasterize (&source_pih, false, false);
        rtds.m_rtss->warp (xf_out, &fixed_pih);

        /* Loop through structures for this atlas image */
        printf ("Vote...\n");
        for (size_t i = 0; i < rtds.m_rtss->get_num_structures(); i++) {
            /* Check structure name, make sure it is something we 
               want to segment */
            std::string ori_name = rtds.m_rtss->get_structure_name (i);
            std::string mapped_name = this->map_structure_name (parms, 
                ori_name);
            if (mapped_name == "") {
                continue;
            }

            /* Make a new voter if needed */
            lprintf ("Voting structure %s\n", mapped_name.c_str());
            Mabs_vote *vote;
            std::map<std::string, Mabs_vote*>::const_iterator vote_it 
                = d_ptr->vote_map.find (mapped_name);
            if (vote_it == d_ptr->vote_map.end()) {
                vote = new Mabs_vote;
                d_ptr->vote_map[mapped_name] = vote;
                vote->set_fixed_image (regd.fixed_image->itk_float());
            } else {
                vote = vote_it->second;
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
    for (std::map<std::string, Mabs_vote*>::const_iterator vote_it 
             = d_ptr->vote_map.begin(); 
         vote_it != d_ptr->vote_map.end(); vote_it++)
    {
        Mabs_vote *vote = vote_it->second;
        lprintf ("Normalizing votes\n");
        vote->normalize_votes();

        /* Optionally, get the weight image */
        FloatImageType::Pointer wi = vote->get_weight_image ();

        /* Optionally, save the weight files */
        if (parms.debug) {
            lprintf ("Saving weights\n");
            Pstring fn; 
            fn.format ("%s/weight_%s.nrrd", out_dir.c_str(), 
                vote_it->first.c_str());
            itk_image_save (wi, fn.c_str());
        }

        /* Threshold the weight image */
        UCharImageType::Pointer thresh = itk_threshold_above (wi, 0.5);

        /* Optionally, save the thresholded files */
        /* GCS FIX: After we can create the structure set, we'll make 
           this optional */
        lprintf ("Saving thresholded structures\n");
        Pstring fn; 
        fn.format ("%s/label_%s.nrrd", out_dir.c_str(), 
                vote_it->first.c_str());
        itk_image_save (thresh, fn.c_str());

        /* Assemble into structure set */
    }
}
