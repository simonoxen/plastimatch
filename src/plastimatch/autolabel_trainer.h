/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _autolabel_trainer_h_
#define _autolabel_trainer_h_

#include "plm_config.h"

class plastimatch1_EXPORT Autolabel_trainer
{
  public:
    Autolabel_trainer () {}
    ~Autolabel_trainer () {}
  public:
    void load_input_dir (const char* input_dir);
    void set_task (const char* task);
    void save_libsvm (const char* output_libsvm_fn);
};

#endif
