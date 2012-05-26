/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mabs_subject_h_
#define _mabs_subject_h_

class Volume;

class PLMSEGMENT_API Mabs_subject {
public:
    Mabs_subject ();
    ~Mabs_subject ();

public:
    const char* img_fn;
    const char* ss_fn;

    Volume* img;
    Volume* ss;
}

#endif /* #ifndef _mabs_subject_h_ */
