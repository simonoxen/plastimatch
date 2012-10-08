/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mabs_subject_h_
#define _mabs_subject_h_

#include "plmsegment_config.h"
#include "plm_path.h"

class Volume;

class PLMSEGMENT_API Mabs_subject {
public:
    Mabs_subject ();
    ~Mabs_subject ();

public:
    char img_fn[_MAX_PATH];
    char ss_fn[_MAX_PATH];

    Volume* img;
    Volume* ss;

    Mabs_subject* next;
};

class PLMSEGMENT_API Mabs_subject_manager {
public:
    Mabs_subject_manager ();
    ~Mabs_subject_manager ();

    Mabs_subject* add ();         /* add new subject @ head */
    bool remove (Mabs_subject*);  /* remove specific subject */
    void remove_all ();           /* remove all subjects */
    void select_head ();          /* reset manager selection to head */
    bool select (Mabs_subject*);  /* explictly set manager select */
    Mabs_subject* next ();        /* select next subject in list */
    Mabs_subject* current ();     /* get currently selected subject */

private:
    Mabs_subject* head;     /* head of subject list */
    Mabs_subject* sel;      /* current selected working node */
};

#endif /* #ifndef _mabs_subject_h_ */
