/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mabs_h_
#define _mabs_h_

#include "plmsegment_config.h"
#include <string>

class Mabs_private;
class Mabs_parms;

class PLMSEGMENT_API Mabs {
public:
    Mabs ();
    ~Mabs ();
public:
    Mabs_private *d_ptr;

protected:
    std::string map_structure_name (
        const std::string& ori_name);
    void sanity_checks ();
    void load_atlas_dir_list ();
    void run_single_registration ();
    void run_registration ();
    void segmentation_vote (
        const std::string& registration_id, 
        const std::string& atlas_id, 
        float rho, 
        float sigma);
    void segmentation_label ();
    void run_segmentation ();

public:
    void set_parms (const Mabs_parms *parms);
    void parse_registration_dir (void);

    void prep (const std::string& input_dir, const std::string& output_dir);
    void atlas_prep ();
    void run ();
    void train ();
};

#endif
