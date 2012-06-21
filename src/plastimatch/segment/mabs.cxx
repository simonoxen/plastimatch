/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsegment_config.h"
#include <list>
#include <stdio.h>
#include <stdlib.h>

#include "dir_list.h"
#include "file_util.h"
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
    Rtds rtds;

    /* Do a few sanity checks */
    if (!is_directory (parms.atlas_dir)) {
        print_and_exit ("Atlas dir (%s) is not a directory\n",
            parms.atlas_dir);
    }
    if (!file_exists (parms.registration_config)) {
        print_and_exit ("Couldn't find registration config (%s)\n", 
            parms.registration_config);
    }

    /* Load the labeling file.  For now, we'll assume this is successful. */
    Plm_image pli (parms.labeling_input_fn);

    // create the data structure for saving the automatically generated labels
    
    /* Parse the atlas directory */
    Dir_list d (parms.atlas_dir);
    for (int i = 0; i < d.num_entries; i++) {

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
        regd.fixed_image = &pli;
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
        Plm_image_header source_pih;
        rtds.m_rtss->prune_empty();
        Rtss_structure_set *cxt = rtds.m_rtss->m_cxt;
        source_pih.set_from_gpuit (cxt->rast_dim, cxt->rast_offset, 
            cxt->rast_spacing, 0);
        rtds.m_rtss->rasterize (&source_pih, false, false);
        rtds.m_rtss->warp (xf_out, &fixed_pih);

        /* Voting loop */
        printf ("Vote...\n");
        std::vector<Mabs_vote*> vote_list;
        for (size_t i = 0; i < rtds.m_rtss->get_num_structures(); i++) 
        {
            /* TODO: Check structure name - for now we just assume 
               structures are in the same order for each atlas */

            /* Make a new voter if needed */
            Mabs_vote *vote;
            if (i >= vote_list.size()) {
                vote = new Mabs_vote;
                vote_list.push_back (vote);
            } else {
                vote = vote_list[i];
            }

            /* Extract binary structure as image */
            UCharImageType::Pointer structure_image 
                = rtds.m_rtss->get_structure_image (i);

            /* Send to voter */
#if defined (commentout)
            vote->vote_contribution (
                regd.fixed_image, 
                warped_image, 
                Plm_image (structure_image), 
                like0, 
                like1
            );
#endif
        }

        // each structure in this patient's structures
        // for () {
          // add this structure to the set of structures that we're segmenting
          // atlas_structure = get the binary structure from the label map
          // target_structure += vote(target, atlas_image, atlas_structure,
          // like0, like1)
        // }

        /* *** THRESHOLING *** */
        // for each of the structures we're trying to segment
        // for () {
          // do the final thresholding (or whatnot)
          // }
        // create the final labelmap
        
        /* Don't let regd destructor delete our fixed image */
        regd.fixed_image = 0;

        /* Clean up */
        delete xf_out;
    }
}
