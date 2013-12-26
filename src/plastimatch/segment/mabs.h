/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mabs_h_
#define _mabs_h_

#include "plmsegment_config.h"
#include <string>
#include "itk_image.h"

class Mabs_private;
class Mabs_parms;

class PLMSEGMENT_API Mabs {
public:
    Mabs ();
    ~Mabs ();
public:
    Mabs_private *d_ptr;

protected:
    void sanity_checks ();
    void load_process_dir_list (const std::string& dir);
    void convert (const std::string& input_dir, const std::string& output_dir);
    void prealign (const std::string& input_dir, 
        const std::string& output_dir);
    FloatImageType::Pointer compute_dmap (
        UCharImageType::Pointer& structure_image,
        const std::string& curr_output_dir,
        const std::string& mapped_name);
    void run_registration_loop ();
    void run_single_registration ();
    void run_segmentation ();
    void run_segmentation_loop ();
    void segmentation_vote (const std::string& atlas_id);
    void segmentation_label ();
    void train_internal (bool registration_only);

public:
    void set_parms (const Mabs_parms *parms);
    void parse_registration_dir (const std::string& registration_config);

    void set_segment_input (const std::string& input_fn);
    void set_segment_output_dicom (const std::string& output_dicom_dir);

    void atlas_selection ();
    void atlas_convert ();
    void atlas_prealign ();
    void train_registration ();
    void train ();
    void segment ();
};

#endif
