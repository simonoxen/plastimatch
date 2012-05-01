/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _autolabel_thumbnailer_h_
#define _autolabel_thumbnailer_h_

#include "plm_config.h"
#include "dlib_trainer.h"

class Plm_image;
class Thumbnail;

class Autolabel_thumbnailer {
public:
    Autolabel_thumbnailer ();
    ~Autolabel_thumbnailer ();
public:
    Plm_image *pli;
    Thumbnail *thumb;
public:
    void set_input_image (const char* fn);
    Dlib_trainer::Dense_sample_type make_sample (float slice_loc);
};

#endif