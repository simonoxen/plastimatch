/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mabs_parms_h_
#define _mabs_parms_h_

class Mabs_subject;

class PLMSEGMENT_API Mabs_Parms {
public:
    Mabs_Parms ();
    ~Mabs_Parms ();

    bool parse_args (int argc, char** argv);

private:
    void parse_config (const char* config_fn);
    int set_key_val (const char* key, const char* val, int section);

public:
    /* [TRAINING] */
    const char* atlas_dir;
    const char* training_dir;

    /* [REGISTRATION] */
    const char* registration_config;

    /* [SUBJECT] */
    Mabs_subject* subject_list;    

    /* [STRUCTURES] */
    // to be implemented

    /* [LABELING] */
    const char* labeling_input_fn;
    const char* labeling_output_fn;

    /* misc */
    bool debug;
}

};
#endif /* #ifndef _mabs_parms_h_ */
