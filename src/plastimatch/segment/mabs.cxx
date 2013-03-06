/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsegment_config.h"
#include <list>
#include <stdio.h>
#include <stdlib.h>

#include "dir_list.h"
#include "dice_statistics.h"
#include "distance_map.h"
#include "file_util.h"
#include "hausdorff_distance.h"
#include "itk_adjust.h"
#include "itk_image_save.h"
#include "itk_threshold.h"
#include "logfile.h"
#include "mabs.h"
#include "mabs_parms.h"
#include "mabs_vote.h"
#include "option_range.h"
#include "path_util.h"
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
    /* These are the input parameters */
    const Mabs_parms *parms;
    
    /* atlas_dir_list is a list of the input directories where the 
       original (dicom) atlas files live, one directory per case */
    std::list<std::string> atlas_dir_list;
    /* outdir_base is the output directory when we are 
       doing a labeling task (i.e. not training)
    */
    std::string outdir_base;
    /* traindir_base is the output directory when we are 
       doing a training task (i.e. not labeling), and is of the form:
       training/
    */
    std::string traindir_base;
    /* registration_list is the list of registration parameter files */
    std::list<std::string> registration_list;
    /* output_dir is ??? */
    std::string output_dir;
    /* segmentation_training_dir is of the form:
       training/0035/segmentations/reg_parm.txt/rho_0.5_sigma_1.2/
    */
    std::string segmentation_training_dir;

    /* There is one reference image at a time, which is the 
       image we are segmenting. */
    std::string ref_id;
    Rtds ref_rtds;
    std::list<std::string> atlas_list;

    /* Segmentation parameters */
    std::string registration_id;
    float minsim;
    float rho;
    float sigma;

    /* While segmenting an image, we sometimes loop through 
       the structures for evaluation.  This holds the 
       binary image of the current structure which we are 
       evaluating. */
    bool have_ref_structure;
    UCharImageType::Pointer ref_structure_image;

    /* You can set these variables to save some intermediate data 
       for debugging and tuning */
    bool compute_distance_map;
    bool write_weight_files;
    bool write_thresholded_files;
    bool write_distance_map;
    bool write_registration_files;

    /* While looping through atlases, the voting information is stored here */
    std::map<std::string, Mabs_vote*> vote_map;

    /* Store timing information for performance evaluation */
    double time_dice;
    double time_dmap;
    double time_extract;
    double time_hausdorff;
    double time_io;
    double time_reg;
    double time_vote;
    double time_warp_img;
    double time_warp_str;

public:
    Mabs_private () {
        parms = 0;
        compute_distance_map = true;
        write_weight_files = false;
        write_thresholded_files = true;
        write_distance_map = true;
        write_registration_files = true;
        have_ref_structure = false;
        this->reset_timers ();

        registration_id = "";
        minsim = 0.0;
        rho = 0.0;
        sigma = 0.0;
    }
    void reset_timers () {
        time_dice = 0;
        time_dmap = 0;
        time_extract = 0;
        time_hausdorff = 0;
        time_io = 0;
        time_reg = 0;
        time_vote = 0;
        time_warp_img = 0;
        time_warp_str = 0;
    }
    void clear_vote_map () {
        std::map<std::string, Mabs_vote*>::iterator it;
        for (it = vote_map.begin(); it != vote_map.end(); ++it) {
            delete it->second;
        }
        vote_map.clear ();
    }
public:
    std::string map_structure_name (const std::string& ori_name);
    void extract_reference_image (const std::string& mapped_name);
    void segmentation_threshold_weight (
        FloatImageType::Pointer weight_image, 
        const std::string& mapped_name, 
        const std::string& structure_label, 
        float thresh_val);
};

/* Map an input-specific structure name to a canonical structure name 
   Return "" if no canonical name */
std::string
Mabs_private::map_structure_name (
    const std::string& ori_name)
{
    if (this->parms->structure_map.size() == 0) {
        lprintf (" > no structure list specified\n");
        return ori_name;
    }

    std::map<std::string, std::string>::const_iterator it 
        = this->parms->structure_map.find (ori_name);
    if (it == this->parms->structure_map.end()) {
        lprintf (" > irrelevant structure: %s\n", ori_name.c_str());
        return "";
    }

    const std::string& mapped_name = it->second;
    if (mapped_name == "") {
        lprintf (" > irrelevant structure: %s\n", ori_name.c_str());
    }
    else if (mapped_name == ori_name) {
        lprintf (" > relevant structure: %s\n", ori_name.c_str());
    }
    else {
        lprintf (" > relevant structure: %s -> %s\n", 
            ori_name.c_str(), mapped_name.c_str());
    }
    return mapped_name;
}

/* Extract reference structure as binary mask, and save into the 
   variable d_ptr->ref_structure_image.
   This image is used when computing dice statistics. 

   GCS FIX: This is inefficient, it could be extracted 
   once at the beginning, and cached. */
void
Mabs_private::extract_reference_image (const std::string& mapped_name)
{
    this->have_ref_structure = false;
    if (this->ref_rtds.m_rtss){
        for (size_t j = 0; 
             j < this->ref_rtds.m_rtss->get_num_structures(); 
             j++)
        {
            std::string ref_ori_name 
                = this->ref_rtds.m_rtss->get_structure_name (j);
            std::string ref_mapped_name 
                = this->map_structure_name (ref_ori_name);
            if (ref_mapped_name == mapped_name) {
                this->ref_structure_image = this->ref_rtds.m_rtss
                    ->get_structure_image (j);
                this->have_ref_structure = true;
                break;
            }
        }
    }
}

void
Mabs_private::segmentation_threshold_weight (
    FloatImageType::Pointer weight_image, 
    const std::string& mapped_name, 
    const std::string& structure_label, 
    float thresh_val
)
{
    Plm_timer timer;

    /* Threshold the weight image */
    timer.start();
    UCharImageType::Pointer thresh_img = itk_threshold_above (
        weight_image, thresh_val);
    this->time_vote += timer.report();

    /* Optionally, save the thresholded files */
    if (this->write_thresholded_files) {
        lprintf ("Saving thresholded structures\n");
        std::string thresh_img_fn = string_format (
            "%s/%s_thresh_%f.nrrd", 
            this->segmentation_training_dir.c_str(), 
            structure_label.c_str(), 
            thresh_val);
        timer.start();
        itk_image_save (thresh_img, thresh_img_fn.c_str());
        this->time_io += timer.report();
    }

    /* Extract reference structure as binary mask. */
    timer.start();
    this->extract_reference_image (mapped_name);
    this->time_extract += timer.report();

    /* Compute Dice, etc. */
    if (this->have_ref_structure) {
        timer.start();
        lprintf ("Computing Dice...\n");
        Dice_statistics dice;
        dice.set_reference_image (this->ref_structure_image);
        dice.set_compare_image (thresh_img);
        dice.run ();
        this->time_dice += timer.report();

        timer.start();
        lprintf ("Computing Hausdorff...\n");
        Hausdorff_distance hd;
        hd.set_reference_image (this->ref_structure_image);
        hd.set_compare_image (thresh_img);
        hd.run ();
        this->time_hausdorff += timer.report();

        std::string seg_log_string = string_format (
            "%s,%s,%s,"
            "rho=%f,sigma=%f,minsum=%f,thresh=%f,"
            "dice=%f,tp=%d,tn=%d,fp=%d,fn=%d,"
            "hd=%f,95hd=%f,ahd=%f,"
            "bhd=%f,95bhd=%f,abhd=%f\n",
            this->ref_id.c_str(),
            this->registration_id.c_str(),
            mapped_name.c_str(),
            this->rho,
            this->sigma,
            this->minsim,
            thresh_val, 
            dice.get_dice(),
            (int) dice.get_true_positives(),
            (int) dice.get_true_negatives(),
            (int) dice.get_false_positives(),
            (int) dice.get_false_negatives(),
            hd.get_hausdorff(),
            hd.get_percent_hausdorff(),
            hd.get_average_hausdorff(),
            hd.get_boundary_hausdorff(),
            hd.get_percent_boundary_hausdorff(),
            hd.get_average_boundary_hausdorff()
        );
        lprintf ("%s", seg_log_string.c_str());

        /* Update seg_dice file */
        std::string seg_dice_log_fn = string_format (
            "%s/seg_dice.csv",
            this->traindir_base.c_str());
        FILE *fp = fopen (seg_dice_log_fn.c_str(), "a");
        fprintf (fp, "%s", seg_log_string.c_str());
        fclose (fp);
    }
}

Mabs::Mabs () {
    d_ptr = new Mabs_private;
}

Mabs::~Mabs () {
    delete d_ptr;
}

void
Mabs::sanity_checks ()
{
    /* Do a few sanity checks */
    if (!is_directory (d_ptr->parms->atlas_dir)) {
        print_and_exit ("Atlas dir (%s) is not a directory\n",
            d_ptr->parms->atlas_dir.c_str());
    }
    if (!is_directory (d_ptr->parms->registration_config)) {
        if (!file_exists (d_ptr->parms->registration_config)) {
            print_and_exit ("Couldn't find registration config (%s)\n", 
                d_ptr->parms->registration_config.c_str());
        }
    }

    /* Make sure there is an output directory */
    d_ptr->outdir_base = d_ptr->parms->labeling_output_fn;
    if (d_ptr->outdir_base == "") {
        d_ptr->outdir_base = "mabs";
    }

    /* Make sure there is a training directory */
    d_ptr->traindir_base = d_ptr->parms->training_dir;
    if (d_ptr->traindir_base == "") {
        d_ptr->traindir_base = "training";
    }
}

void
Mabs::load_atlas_dir_list ()
{
    Dir_list d (d_ptr->parms->atlas_dir);
    for (int i = 0; i < d.num_entries; i++)
    {
        /* Skip "." and ".." */
        if (!strcmp (d.entries[i], ".") || !strcmp (d.entries[i], "..")) {
            continue;
        }

        /* Build string containing full path to atlas item */
        std::string path = compose_filename (
            d_ptr->parms->atlas_dir, d.entries[i]);

        /* Only consider directories */
        if (!is_directory (path.c_str())) {
            continue;
        }

        /* Add directory to atlas_dir_list */
        d_ptr->atlas_dir_list.push_back (path);
    }
}

void
Mabs::prep (const std::string& input_dir, const std::string& output_dir)
{
    Rtds rtds;
    Plm_timer timer;

    /* Load the rtds for the atlas */
    timer.start();
    lprintf ("MABS loading %s\n", input_dir.c_str());
    rtds.load_dicom_dir (input_dir.c_str());
    d_ptr->time_io += timer.report();

    /* Save the image as raw files */
    timer.start();
    std::string fn = string_format ("%s/img.nrrd", output_dir.c_str());
    rtds.m_img->save_image (fn.c_str());

    /* Remove structures which are not part of the atlas */
    timer.start();
    rtds.m_rtss->prune_empty ();
    Rtss_structure_set *cxt = rtds.m_rtss->m_cxt;
    for (size_t i = 0; i < rtds.m_rtss->get_num_structures(); i++) {
        /* Check structure name, make sure it is something we 
           want to segment */
        std::string ori_name = rtds.m_rtss->get_structure_name (i);
        std::string mapped_name = d_ptr->map_structure_name (ori_name);
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

    /* Save structures which are part of the atlas */
    std::string prefix = string_format ("%s/structures", output_dir.c_str());
    rtds.m_rtss->save_prefix (prefix, "nrrd");
    d_ptr->time_io += timer.report();
}

void
Mabs::atlas_prep ()
{
    /* Do a few sanity checks */
    this->sanity_checks ();

    /* Parse atlas directory */
    this->load_atlas_dir_list ();

    /* Loop through atlas_dir, converting file formats */
    for (std::list<std::string>::iterator it = d_ptr->atlas_dir_list.begin();
         it != d_ptr->atlas_dir_list.end(); it++)
    {
        std::string input_dir = *it;
        std::string atlas_id = basename (input_dir);
        std::string output_dir = string_format (
            "%s/atlas/%s", d_ptr->traindir_base.c_str(), 
            atlas_id.c_str());

        this->prep (input_dir, output_dir);
    }
    lprintf ("Rasterization time:   %10.1f seconds\n", d_ptr->time_extract);
    lprintf ("I/O time:             %10.1f seconds\n", d_ptr->time_io);
    lprintf ("MABS prep complete\n");
}

void
Mabs::parse_registration_dir (void)
{
    /* Figure out whether we need to do a single registration 
       or multiple registrations (for atlas tuning) */

    if (is_directory (d_ptr->parms->registration_config)) {
        Dir_list dir (d_ptr->parms->registration_config);
        for (int i = 0; i < dir.num_entries; i++) {
            std::string full_path = string_format (
                "%s/%s", d_ptr->parms->registration_config.c_str(), 
                dir.entries[i]);
            /* Skip backup files */
            if (extension_is (dir.entries[i], "~")) {
                continue;
            }
            /* Skip directories */
            if (is_directory (full_path)) {
                continue;
            }
            d_ptr->registration_list.push_back (full_path);
        }
    }
    else {
        d_ptr->registration_list.push_back (d_ptr->parms->registration_config);
    }
}

void
Mabs::run_registration ()
{
    Plm_timer timer;

    /* Loop through images in the atlas */
    std::list<std::string>::iterator atl_it;
    for (atl_it = d_ptr->atlas_list.begin();
         atl_it != d_ptr->atlas_list.end(); atl_it++)
    {
        Rtds rtds;
        std::string path = *atl_it;
        std::string atlas_id = basename (path);
        std::string atlas_input_path = string_format ("%s/atlas/%s",
            d_ptr->traindir_base.c_str(), atlas_id.c_str());
        std::string atlas_output_path = string_format ("%s/%s",
            d_ptr->output_dir.c_str(), atlas_id.c_str());

        /* Check if this registration is already complete.
           We might be able to skip it. */
        std::string atl_checkpoint_fn = string_format (
            "%s/checkpoint.txt", atlas_output_path.c_str());
        if (file_exists (atl_checkpoint_fn)) {
            lprintf ("Atlas registration complete for %s\n",
                atlas_output_path.c_str());
            continue;
        }

        /* Load image & structures from "prep" directory */
        timer.start();
        std::string fn = string_format ("%s/img.nrrd", 
            atlas_input_path.c_str());
        rtds.m_img = plm_image_load_native (fn.c_str());
        fn = string_format ("%s/structures", 
            atlas_input_path.c_str());
        rtds.m_rtss = new Rtss;
        rtds.m_rtss->load_prefix (fn.c_str());
        d_ptr->time_io += timer.report();

        /* Inspect the structures -- we might be able to skip the 
           atlas if it has no relevant structures */
        bool can_skip = true;
        for (size_t i = 0; i < rtds.m_rtss->get_num_structures(); i++) {
            std::string ori_name = rtds.m_rtss->get_structure_name (i);
            std::string mapped_name = d_ptr->map_structure_name (ori_name);
            if (mapped_name != "") {
                can_skip = false;
                break;
            }
        }
        if (can_skip) {
            lprintf ("No relevant structures. Skipping.\n");
            continue;
        }

        /* Loop through each registration parameter set */
        std::list<std::string>::iterator reg_it;
        for (reg_it = d_ptr->registration_list.begin(); 
             reg_it != d_ptr->registration_list.end(); reg_it++) 
        {
            /* Set up files & directories for this job */
            std::string command_file = *reg_it;
            std::string curr_output_dir;
            std::string registration_id;
            registration_id = basename (command_file);
            curr_output_dir = string_format ("%s/%s",
                atlas_output_path.c_str(),
                registration_id.c_str());

            /* Check if this registration is already complete.
               We might be able to skip it. */
            std::string reg_checkpoint_fn = string_format (
                "%s/checkpoint.txt", curr_output_dir.c_str());
            if (file_exists (reg_checkpoint_fn)) {
                lprintf ("Registration parms complete for %s\n",
                    curr_output_dir.c_str());
                continue;
            }

            /* Make a registration command string */
            lprintf ("Processing command file: %s\n", command_file.c_str());
            std::string command_string = slurp_file (command_file);

            /* Parse the registration command string */
            Registration_parms *regp = new Registration_parms;
            int rc = regp->set_command_string (command_string);
            if (rc) {
                lprintf ("Skipping command file \"%s\" "
                    "due to parse error.\n", command_file.c_str());
                delete regp;
                continue;
            }

            /* Manually set input files */
            Registration_data *regd = new Registration_data;
            regd->fixed_image = d_ptr->ref_rtds.m_img;
            regd->moving_image = rtds.m_img;

            /* Run the registration */
            Xform *xf_out;
            lprintf ("DO_REGISTRATION_PURE\n");
            lprintf ("regp->num_stages = %d\n", regp->num_stages);
            timer.start();
            do_registration_pure (&xf_out, regd, regp);
            d_ptr->time_reg += timer.report();

            /* Warp the output image */
            lprintf ("Warp output image...\n");
            Plm_image_header fixed_pih (regd->fixed_image);
            Plm_image *warped_image = new Plm_image;
            timer.start();
            plm_warp (warped_image, 0, xf_out, &fixed_pih, regd->moving_image, 
                regp->default_value, 0, 1);
            d_ptr->time_warp_img += timer.report();
            
            /* We're done with this */
            delete regp;

            /* Warp the structures */
            lprintf ("Warp structures...\n");
            Plm_image_header source_pih (rtds.m_img);
            timer.start();
            rtds.m_rtss->warp (xf_out, &fixed_pih);
            d_ptr->time_warp_str += timer.report();

            /* Save some debugging information */
            if (d_ptr->write_registration_files) {
                timer.start();
                lprintf ("Saving registration_files\n");
                std::string fn;
                fn = string_format ("%s/img.nrrd", curr_output_dir.c_str());
                warped_image->save_image (fn.c_str());

                fn = string_format ("%s/xf.txt", curr_output_dir.c_str());
                xf_out->save (fn.c_str());

                fn = string_format ("%s/structures", curr_output_dir.c_str());
                rtds.m_rtss->save_prefix (fn, "nrrd");
                d_ptr->time_io += timer.report();
            }

            /* We're done with these */
            delete regd;
            delete warped_image;

            /* Loop through structures for this atlas image */
            lprintf ("Process structures...\n");
            for (size_t i = 0; i < rtds.m_rtss->get_num_structures(); i++) {
                /* Check structure name, make sure it is something we 
                   want to segment */
                std::string ori_name = rtds.m_rtss->get_structure_name (i);
                std::string mapped_name = d_ptr->map_structure_name (ori_name);
                if (mapped_name == "") {
                    continue;
                }

                /* Extract structure as binary mask */
                timer.start();
                UCharImageType::Pointer structure_image 
                    = rtds.m_rtss->get_structure_image (i);
                d_ptr->time_extract += timer.report();

                /* Make the distance map */
                timer.start();
                lprintf ("Computing distance map...\n");
                Distance_map dmap;
                if (d_ptr->compute_distance_map) {
                    dmap.set_input_image (structure_image);
                    dmap.run ();
                    FloatImageType::Pointer dmap_image 
                        = dmap.get_output_image ();
                    d_ptr->time_dmap += timer.report();

                    /* Truncate the dmap.  This is to save disk space. 
                       Maybe we won't need this if we can crop. */
                    Adjustment_list al;
                    al.push_back (std::make_pair (
                            -std::numeric_limits<float>::max(), 0));
                    al.push_back (std::make_pair (-400, -400));
                    al.push_back (std::make_pair (400, 400));
                    al.push_back (std::make_pair (
                            std::numeric_limits<float>::max(), 0));
                    itk_adjust (dmap_image, al);
                    if (d_ptr->write_distance_map) {
                        timer.start();
                        fn = string_format ("%s/dmap_%s.nrrd", 
                            curr_output_dir.c_str(), mapped_name.c_str());
                        itk_image_save (dmap_image, fn.c_str());
                        d_ptr->time_io += timer.report();
                    }
                }

                /* Extract reference structure as binary mask. */
                timer.start();
                d_ptr->extract_reference_image (mapped_name);
                d_ptr->time_extract += timer.report();

                /* Compute Dice, etc. */
                timer.start();
                if (d_ptr->have_ref_structure) {
                    lprintf ("Computing Dice...\n");
                    Dice_statistics dice;
                    dice.set_reference_image (d_ptr->ref_structure_image);
                    dice.set_compare_image (structure_image);
                    dice.run ();

                    std::string reg_log_string = string_format (
                        "%s,%s,%s,%s,%f,%d,%d,%d,%d\n",
                        d_ptr->ref_id.c_str(), 
                        atlas_id.c_str(),
                        registration_id.c_str(),
                        mapped_name.c_str(), 
                        dice.get_dice(),
                        (int) dice.get_true_positives(),
                        (int) dice.get_true_negatives(),
                        (int) dice.get_false_positives(),
                        (int) dice.get_false_negatives());
                    lprintf ("%s", reg_log_string.c_str());

                    /* Update reg_dice file */
                    std::string reg_dice_log_fn = string_format (
                        "%s/reg_dice.csv",
                        d_ptr->traindir_base.c_str());
                    FILE *fp = fopen (reg_dice_log_fn.c_str(), "a");
                    fprintf (fp, "%s", reg_log_string.c_str());
                    fclose (fp);
                }
                d_ptr->time_dice += timer.report();
            }

            /* Create checkpoint file which means that this registration
               is complete */
            touch_file (reg_checkpoint_fn);

            /* Clean up */
            delete xf_out;
        } /* end for each registration parameter */

        /* Create checkpoint file which means that training for 
           this atlas example is complete */
        touch_file (atl_checkpoint_fn);
    } /* end for each atlas image */
}

void
Mabs::segmentation_vote (const std::string& atlas_id)
{
    Plm_timer timer;
    
    /* Set up files & directories for this job */
    std::string atlas_output_path;
    atlas_output_path = string_format ("%s/%s",
        d_ptr->output_dir.c_str(), atlas_id.c_str());
    lprintf ("atlas_output_path: %s, %s, %s\n",
        d_ptr->output_dir.c_str(), atlas_id.c_str(),
        atlas_output_path.c_str());
    std::string curr_output_dir;
    curr_output_dir = string_format ("%s/%s",
        atlas_output_path.c_str(),
        d_ptr->registration_id.c_str());
    lprintf ("curr_output_dir: %s\n", curr_output_dir.c_str());

    /* Load warped image */
    timer.start();
    std::string warped_image_fn;
    warped_image_fn = string_format (
        "%s/img.nrrd", curr_output_dir.c_str());
    lprintf ("Loading warped image: %s\n", warped_image_fn.c_str());
    Plm_image *warped_image = plm_image_load_native (
        warped_image_fn);
    d_ptr->time_io += timer.report();
            
    /* Loop through structures for this atlas image */
    std::map<std::string, std::string>::const_iterator it;
    for (it = d_ptr->parms->structure_map.begin ();
         it != d_ptr->parms->structure_map.end (); it++)
    {
        std::string mapped_name = it->first;
        lprintf ("Segmenting structure: %s\n", mapped_name.c_str());

        /* Make a new voter if needed */
        Mabs_vote *vote;
        std::map<std::string, Mabs_vote*>::const_iterator vote_it 
            = d_ptr->vote_map.find (mapped_name);
        if (vote_it == d_ptr->vote_map.end()) {
            vote = new Mabs_vote;
            vote->set_rho (d_ptr->rho);
            vote->set_sigma (d_ptr->sigma);
            vote->set_minimum_similarity (d_ptr->minsim);
            d_ptr->vote_map[mapped_name] = vote;
            vote->set_fixed_image (
                d_ptr->ref_rtds.m_img->itk_float());
        } else {
            vote = vote_it->second;
        }

        /* Load dmap */
        timer.start();
        lprintf ("Loading dmap\n");
        std::string dmap_fn = string_format ("%s/dmap_%s.nrrd", 
            curr_output_dir.c_str(), mapped_name.c_str());
        Plm_image *dmap_image = plm_image_load_native (
            dmap_fn.c_str());
        d_ptr->time_io += timer.report();
        if (!dmap_image) {
            continue;
        }

        /* Vote */
        timer.start();
        lprintf ("Voting\n");
        vote->vote (warped_image->itk_float(), 
            dmap_image->itk_float());
        d_ptr->time_vote += timer.report();

        /* We don't need this any more */
        delete dmap_image;
    }

    /* We don't need this any more */
    delete warped_image;
}

void
Mabs::segmentation_label ()
{
    Plm_timer timer;

    /* Set up files & directories for this job */
    d_ptr->segmentation_training_dir
        = string_format ("%s/segmentations/%s/rho_%f_sig_%f_ms_%f",
            d_ptr->output_dir.c_str(), d_ptr->registration_id.c_str(),
            d_ptr->rho, d_ptr->sigma, d_ptr->minsim);
    lprintf ("segmentation_training_dir: %s\n", 
        d_ptr->segmentation_training_dir.c_str());

    /* Get output image for each label */
    lprintf ("Normalizing and saving weights\n");
    for (std::map<std::string, Mabs_vote*>::const_iterator vote_it 
             = d_ptr->vote_map.begin(); 
         vote_it != d_ptr->vote_map.end(); vote_it++)
    {
        const std::string& mapped_name = vote_it->first;
        Mabs_vote *vote = vote_it->second;
        lprintf ("Normalizing votes\n");
        timer.start();
        vote->normalize_votes();
        d_ptr->time_vote += timer.report();

        /* Get the weight image */
        FloatImageType::Pointer weight_image;
        weight_image = vote->get_weight_image ();

        /* Optionally, save the weight files */
        if (d_ptr->write_weight_files) {
            lprintf ("Saving weights\n");
            std::string fn = string_format ("%s/weight_%s.nrrd", 
                d_ptr->segmentation_training_dir.c_str(),
                vote_it->first.c_str());
            timer.start();
            itk_image_save (weight_image, fn.c_str());
            d_ptr->time_io += timer.report();
        }

        Option_range thresh_range;
        thresh_range.set_range (d_ptr->parms->threshold_values);

        /* Loop through each threshold value, do thresholding,
           and then record score */

        const std::list<float>& thresh_list = thresh_range.get_range();
        std::list<float>::const_iterator thresh_it;
        for (thresh_it = thresh_list.begin(); 
             thresh_it != thresh_list.end(); thresh_it++) 
        {
            d_ptr->segmentation_threshold_weight (
                weight_image, mapped_name, 
                vote_it->first.c_str(), *thresh_it);
        }
    }
}

void
Mabs::run_segmentation ()
{
    /* Clear out internal structure */
    d_ptr->clear_vote_map ();

    /* Check if this segmentation is already complete.
       We might be able to skip it. */
    std::string curr_output_dir;
    curr_output_dir = string_format (
        "%s/segmentations/%s/rho_%f_sig_%f_ms_%f",
        d_ptr->output_dir.c_str(), d_ptr->registration_id.c_str(),
        d_ptr->rho, d_ptr->sigma, d_ptr->minsim);
    std::string seg_checkpoint_fn = string_format (
        "%s/checkpoint.txt", curr_output_dir.c_str());
    if (file_exists (seg_checkpoint_fn)) {
        lprintf ("Segmentation complete for %s\n",
            curr_output_dir.c_str());
        return;
    }

    /* Loop through images in the atlas */
    std::list<std::string>::iterator atl_it;
    for (atl_it = d_ptr->atlas_list.begin();
         atl_it != d_ptr->atlas_list.end(); atl_it++)
    {
        std::string atlas_id = basename (*atl_it);
        segmentation_vote (atlas_id);
    }

    /* Threshold images based on weight */
    segmentation_label ();

    /* Create checkpoint file which means that this segmentation
       is complete */
    touch_file (seg_checkpoint_fn);

    /* Clear out internal structure */
    d_ptr->clear_vote_map ();
}


void
Mabs::run_segmentation_loop ()
{
    Option_range minsim_range, rho_range, sigma_range;
    minsim_range.set_range (d_ptr->parms->minsim_values);
    rho_range.set_range (d_ptr->parms->rho_values);
    sigma_range.set_range (d_ptr->parms->sigma_values);

    /* Loop through each registration parameter set */
    std::list<std::string>::iterator reg_it;
    for (reg_it = d_ptr->registration_list.begin(); 
         reg_it != d_ptr->registration_list.end(); reg_it++) 
    {
        d_ptr->registration_id = basename (*reg_it);

        /* Loop through each training parameter: rho */
        const std::list<float>& rho_list = rho_range.get_range();
        std::list<float>::const_iterator rho_it;
        for (rho_it = rho_list.begin(); rho_it != rho_list.end(); rho_it++) 
        {
            d_ptr->rho = *rho_it;

            /* Loop through each training parameter: sigma */
            const std::list<float>& sigma_list = sigma_range.get_range();
            std::list<float>::const_iterator sigma_it;
            for (sigma_it = sigma_list.begin(); 
                 sigma_it != sigma_list.end(); sigma_it++) 
            {
                d_ptr->sigma = *sigma_it;

                /* Loop through each training parameter: minimum similarity */
                const std::list<float>& minsim_list = minsim_range.get_range();
                std::list<float>::const_iterator minsim_it;
                for (minsim_it = minsim_list.begin(); 
                     minsim_it != minsim_list.end(); minsim_it++) 
                {
                    d_ptr->minsim = *minsim_it;

                    run_segmentation ();
                }
            }
        }
    }
}

void 
Mabs::set_parms (const Mabs_parms *parms)
{
    d_ptr->parms = parms;
    d_ptr->write_thresholded_files = parms->write_thresholded_files;
    d_ptr->write_weight_files = parms->write_weight_files;
}

void
Mabs::run ()
{
    /* Do a few sanity checks */
    this->sanity_checks ();

    /* Parse directory with registration files */
    this->parse_registration_dir ();

    /* Load the image to be labeled.  For now, we'll assume this 
       is successful. */
    d_ptr->ref_rtds.m_img = plm_image_load_native (
        d_ptr->parms->labeling_input_fn);

    /* Parse atlas directory */
    this->load_atlas_dir_list ();

    /* Set atlas_list */
    d_ptr->atlas_list = d_ptr->atlas_dir_list;

    /* Set output dir for this test case */
    d_ptr->output_dir = d_ptr->outdir_base;

    /* Save it for debugging */
    std::string fn = string_format ("%s/%s", d_ptr->outdir_base.c_str(),
        "img.nrrd");
    d_ptr->ref_rtds.m_img->save_image (fn.c_str());

    /* Run the segmentation */
    this->run_registration ();
}

void
Mabs::train_internal (bool registration_only)
{
    Plm_timer timer;
    Plm_timer timer_total;
    timer_total.start();

    /* Do a few sanity checks */
    this->sanity_checks ();

    /* Open logfile */
    std::string logfile_path = string_format (
        "%s/%s", d_ptr->traindir_base.c_str(), "logfile.txt");
    logfile_open (logfile_path.c_str());

    /* Parse directory with registration files */
    this->parse_registration_dir ();

    /* Parse atlas directory */
    this->load_atlas_dir_list ();

    /* Loop through atlas_dir, choosing reference images to segment */
    for (std::list<std::string>::iterator it = d_ptr->atlas_dir_list.begin();
         it != d_ptr->atlas_dir_list.end(); it++)
    {
        /* Create atlas list for this test case */
        std::string path = *it;
        d_ptr->atlas_list = d_ptr->atlas_dir_list;
        d_ptr->atlas_list.remove (path);

        /* Set output dir for this test case */
        std::string patient_id = basename (path);
        d_ptr->ref_id = patient_id;
        d_ptr->output_dir = string_format ("%s/%s",
            d_ptr->traindir_base.c_str(), patient_id.c_str());
        lprintf ("outdir = %s\n", d_ptr->output_dir.c_str());

        /* Load image & structures from "prep" directory */
        timer.start();
        std::string fn = string_format ("%s/atlas/%s/img.nrrd", 
            d_ptr->traindir_base.c_str(), patient_id.c_str());
        d_ptr->ref_rtds.m_img = plm_image_load_native (fn.c_str());
        fn = string_format ("%s/atlas/%s/structures", 
            d_ptr->traindir_base.c_str(), patient_id.c_str());
        d_ptr->ref_rtds.m_rtss = new Rtss;
        d_ptr->ref_rtds.m_rtss->load_prefix (fn.c_str());
        d_ptr->time_io += timer.report();

        /* Run the segmentation */
        this->run_registration ();
        if (!registration_only) {
            this->run_segmentation_loop ();
        }
    }

    lprintf ("Registration time:    %10.1f seconds\n", d_ptr->time_reg);
    lprintf ("Warping time (img):   %10.1f seconds\n", d_ptr->time_warp_img);
    lprintf ("Warping time (str):   %10.1f seconds\n", d_ptr->time_warp_str);
    lprintf ("Extraction time:      %10.1f seconds\n", d_ptr->time_extract);
    lprintf ("Dice time:            %10.1f seconds\n", d_ptr->time_dice);
    lprintf ("Hausdorff time:       %10.1f seconds\n", d_ptr->time_hausdorff);
    lprintf ("Distance map time:    %10.1f seconds\n", d_ptr->time_dmap);
    lprintf ("Voting time:          %10.1f seconds\n", d_ptr->time_vote);
    lprintf ("I/O time:             %10.1f seconds\n", d_ptr->time_io);
    lprintf ("Total time:           %10.1f seconds\n", timer_total.report());
    lprintf ("MABS training complete\n");

    logfile_close ();
}

void
Mabs::train ()
{
    d_ptr->compute_distance_map = true;
    d_ptr->write_distance_map = true;
    this->train_internal (false);
}

void
Mabs::train_registration ()
{
    d_ptr->compute_distance_map = false;
    d_ptr->write_distance_map = false;
    this->train_internal (true);
}
