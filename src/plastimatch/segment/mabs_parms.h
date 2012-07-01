/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mabs_parms_h_
#define _mabs_parms_h_

#include "plmsegment_config.h"
#include "plm_path.h"
#include "pstring.h"

class Mabs_subject_manager;

class PLMSEGMENT_API Mabs_parms {
public:
    Mabs_parms ();
    ~Mabs_parms ();

    bool parse_args (int argc, char** argv);
    void print ();

private:
    void parse_config (const char* config_fn);
    int set_key_val (const char* key, const char* val, int section);

public:
    /* [TRAINING] */
    char atlas_dir[_MAX_PATH];
    char training_dir[_MAX_PATH];

    /* [REGISTRATION] */
    char registration_config[_MAX_PATH];

    /* [SUBJECT] */
    Mabs_subject_manager* sman;    

    /* [STRUCTURES] */
    // to be implemented

    /* [LABELING] */
    char labeling_input_fn[_MAX_PATH];
    Pstring labeling_output_fn;

    /* misc */
    bool debug;
};

#endif /* #ifndef _mabs_parms_h_ */
