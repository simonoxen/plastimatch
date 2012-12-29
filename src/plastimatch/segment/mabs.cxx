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
#include "plm_image_header.h"
#include "plm_stages.h"
#include "plm_timer.h"
#include "plm_warp.h"
#include "print_and_exit.h"
#include "registration_data.h"
#include "registration_parms.h"
#include "rtds.h"
#include "rtss.h"
#include "rtss_structure_set.h"
#include "string_util.h"
#include "xform.h"

class Mabs_private {
public:
    std::map<std::string, Mabs_vote*> vote_map;
    std::list<std::string> atlas_dir_list;
    std::string outdir_base;
    std::string traindir_base;

    Plm_image fixed_image;
    std::list<std::string> atlas_list;
    std::string output_dir;
    std::string input_dir;

    bool write_weight_files;
    bool write_registration_files;

    double time_extract;
    double time_io;
    double time_reg;
    double time_vote;
    double time_warp;

public:
    Mabs_private () {
        time_extract = 0;
        time_io = 0;
        time_reg = 0;
        time_vote = 0;
        time_warp = 0;
        write_weight_files = false;
        write_registration_files = true;
    }
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
Mabs::sanity_checks (const Mabs_parms& parms)
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
    d_ptr->outdir_base = parms.labeling_output_fn;
    if (d_ptr->outdir_base == "") {
        d_ptr->outdir_base = "mabs";
    }

    /* Make sure there is a training directory */
    d_ptr->traindir_base = parms.training_dir;
    if (d_ptr->traindir_base == "") {
        d_ptr->traindir_base = "training";
    }
}

void
Mabs::load_atlas_dir_list (const Mabs_parms& parms)
{
    Dir_list d (parms.atlas_dir);
    for (int i = 0; i < d.num_entries; i++)
    {
        /* Skip "." and ".." */
        if (!strcmp (d.entries[i], ".") || !strcmp (d.entries[i], "..")) {
            continue;
        }

        /* Build string containing full path to atlas item */
        std::string path = compose_filename (parms.atlas_dir, d.entries[i]);

        /* Only consider directories */
        if (!is_directory (path.c_str())) {
            continue;
        }

        /* Add directory to atlas_dir_list */
        d_ptr->atlas_dir_list.push_back (path);
    }
}

void
Mabs::run_internal (const Mabs_parms& parms)
{
    Plm_timer timer;

    /* Clear out internal structure */
    d_ptr->vote_map.clear ();

    /* Loop through images in the atlas */
    for (std::list<std::string>::iterator it = d_ptr->atlas_list.begin();
         it != d_ptr->atlas_list.end(); it++)
    {
        Rtds rtds;
        std::string path = *it;

        /* For now, only handle dicom directories.  We assume the 
           load is successful. */
        timer.start();
        lprintf ("MABS loading %s\n", path.c_str());
        rtds.load_dicom_dir (path.c_str());
        rtds.m_rtss->prune_empty ();
        d_ptr->time_io += timer.report();

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
        regd.fixed_image = &d_ptr->fixed_image;
        regd.moving_image = rtds.m_img;

        /* Run the registration */
        Xform *xf_out;
        printf ("DO_REGISTRATION_PURE\n");
        printf ("regp.num_stages = %d\n", regp.num_stages);
        timer.start();
        do_registration_pure (&xf_out, &regd, &regp);
        d_ptr->time_reg += timer.report();

        /* Warp the output image */
        printf ("Warp output image...\n");
        Plm_image_header fixed_pih (regd.fixed_image);
        Plm_image warped_image;
        timer.start();
        plm_warp (&warped_image, 0, xf_out, &fixed_pih, regd.moving_image, 
            regp.default_value, 0, 1);
        d_ptr->time_warp += timer.report();

        /* Save some debugging information */
        if (d_ptr->write_registration_files) {
            lprintf ("Saving registration_files\n");
            std::string tmp_path = strip_leading_dir (path);
            Pstring fn;
            fn.format ("%s/%s/img.nrrd", d_ptr->output_dir.c_str(), 
                tmp_path.c_str());
            warped_image.save_image (fn.c_str());

            fn.format ("%s/%s/xf.txt", d_ptr->output_dir.c_str(), 
                tmp_path.c_str());
            xf_out->save (fn.c_str());
        }

        /* Warp the structures */
        printf ("Warp structures...\n");
        Plm_image_header source_pih (rtds.m_img);
        timer.start();
        rtds.m_rtss->rasterize (&source_pih, false, false);
        rtds.m_rtss->warp (xf_out, &fixed_pih);
        d_ptr->time_warp += timer.report();

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
            timer.start();
            UCharImageType::Pointer structure_image 
                = rtds.m_rtss->get_structure_image (i);
            d_ptr->time_extract += timer.report();

            /* Vote for each voxel */
            timer.start();
            vote->vote (
                warped_image.itk_float(),
                structure_image);
            d_ptr->time_vote += timer.report();
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
        timer.start();
        vote->normalize_votes();
        d_ptr->time_vote += timer.report();

        /* Optionally, get the weight image */
        FloatImageType::Pointer wi = vote->get_weight_image ();

        /* Optionally, save the weight files */
        if (d_ptr->write_weight_files) {
            lprintf ("Saving weights\n");
            Pstring fn; 
            fn.format ("%s/weight_%s.nrrd", d_ptr->output_dir.c_str(), 
                vote_it->first.c_str());
            itk_image_save (wi, fn.c_str());
        }

        /* Threshold the weight image */
        timer.start();
        UCharImageType::Pointer thresh = itk_threshold_above (wi, 0.5);
        d_ptr->time_vote += timer.report();

        /* Optionally, save the thresholded files */
        /* GCS FIX: After we can create the structure set, we'll make 
           this optional */
        lprintf ("Saving thresholded structures\n");
        Pstring fn; 
        fn.format ("%s/label_%s.nrrd", d_ptr->output_dir.c_str(), 
                vote_it->first.c_str());
        timer.start();
        itk_image_save (thresh, fn.c_str());
        d_ptr->time_io += timer.report();

        /* Assemble into structure set */
    }
}

void
Mabs::prep (const Mabs_parms& parms)
{
    /* Do a few sanity checks */
    this->sanity_checks (parms);

    /* Parse atlas directory */
    this->load_atlas_dir_list (parms);

    /* Loop through atlas_dir, converting file formats */
    for (std::list<std::string>::iterator it = d_ptr->atlas_dir_list.begin();
         it != d_ptr->atlas_dir_list.end(); it++)
    {
        Plm_timer timer;
        Rtds rtds;
        std::string path = *it;

        /* Load the rtds */
        timer.start();
        lprintf ("MABS loading %s\n", path.c_str());
        rtds.load_dicom_dir (path.c_str());
        d_ptr->time_io += timer.report();

        /* Save the image as raw files */
        timer.start();
        std::string tmp_path = strip_leading_dir (path);
        std::string fn = string_format ("%s/%s/%s/%s.nrrd", 
            d_ptr->traindir_base.c_str(), tmp_path.c_str(), tmp_path.c_str(),
            tmp_path.c_str());
        rtds.m_img->save_image (fn.c_str());

        /* Remove structures which are not part of the atlas */
        timer.start();
        rtds.m_rtss->prune_empty ();
        Rtss_structure_set *cxt = rtds.m_rtss->m_cxt;
        for (size_t i = 0; i < rtds.m_rtss->get_num_structures(); i++) {
            /* Check structure name, make sure it is something we 
               want to segment */
            std::string ori_name = rtds.m_rtss->get_structure_name (i);
            std::string mapped_name = this->map_structure_name (
                parms, ori_name);
            if (mapped_name == "") {
                /* If not, delete it (before rasterizing) */
                cxt->delete_structure (i);
                --i;
            }
        }

        /* Rasterize structure sets and save */
        Plm_image_header pih (rtds.m_img);
        rtds.m_rtss->rasterize (&pih, false, false);
        d_ptr->time_extract += timer.report();

        /* Save strcutres which are part of the atlas */
        std::string prefix = string_format ("%s/%s/%s/", 
            d_ptr->traindir_base.c_str(), tmp_path.c_str(), tmp_path.c_str());
        rtds.m_rtss->save_prefix (prefix.c_str());
        d_ptr->time_io += timer.report();
    }
    printf ("Rasterization time:   %10.1f seconds\n", d_ptr->time_extract);
    printf ("I/O time:             %10.1f seconds\n", d_ptr->time_io);
    printf ("MABS prep complete\n");
}

void
Mabs::run (const Mabs_parms& parms)
{
    /* Do a few sanity checks */
    this->sanity_checks (parms);

    /* Load the labeling file.  For now, we'll assume this is successful. */
    d_ptr->fixed_image.load_native (parms.labeling_input_fn);

    /* Parse atlas directory */
    this->load_atlas_dir_list (parms);

    /* Set atlas_list */
    d_ptr->atlas_list = d_ptr->atlas_dir_list;

    /* Set output dir for this test case */
    d_ptr->output_dir = d_ptr->outdir_base;

    /* Run the segmentation */
    this->run_internal (parms);
}

void
Mabs::train (const Mabs_parms& parms)
{
    Plm_timer timer;

    /* Do a few sanity checks */
    this->sanity_checks (parms);

    /* Parse atlas directory */
    this->load_atlas_dir_list (parms);

    /* Write some extra files when training */
    d_ptr->write_weight_files = true;

    /* Loop through atlas_dir, doing LOO testing */
    for (std::list<std::string>::iterator it = d_ptr->atlas_dir_list.begin();
         it != d_ptr->atlas_dir_list.end(); it++)
    {
        /* Create atlas list for this test case */
        std::string path = *it;
        d_ptr->atlas_list = d_ptr->atlas_dir_list;
        d_ptr->atlas_list.remove (path);

        /* Set output dir for this test case */
        std::string tmp_path = strip_leading_dir (path);
        d_ptr->output_dir = d_ptr->traindir_base + "/" + tmp_path;
        lprintf ("outdir = %s\n", d_ptr->output_dir.c_str());

        /* Load the input file.  For now, we'll assume this is successful. */
        timer.start();
        d_ptr->fixed_image.load_native (path);
        d_ptr->time_io += timer.report();

        /* Run the segmentation */
        this->run_internal (parms);
    }

    printf ("Registration time:    %10.1f seconds\n", d_ptr->time_reg);
    printf ("Warping time:         %10.1f seconds\n", d_ptr->time_warp);
    printf ("Extraction time:      %10.1f seconds\n", d_ptr->time_extract);
    printf ("Voting time:          %10.1f seconds\n", d_ptr->time_vote);
    printf ("I/O time:             %10.1f seconds\n", d_ptr->time_io);
    printf ("MABS training complete\n");
}
