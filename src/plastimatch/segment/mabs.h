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
        const Mabs_parms& parms, 
        const std::string& ori_name);
    void sanity_checks (const Mabs_parms& parms);
    void load_atlas_dir_list (const Mabs_parms& parms);
    void run_single_registration (const Mabs_parms& parms);
    void run_registration (const Mabs_parms& parms);
    void run_segmentation (const Mabs_parms& parms);

public:
    void prep (const Mabs_parms& parms);
    void run (const Mabs_parms& parms);
    void train (const Mabs_parms& parms);
};

#endif
