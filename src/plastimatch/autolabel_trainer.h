/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _autolabel_trainer_h_
#define _autolabel_trainer_h_

#include "plm_config.h"

class plastimatch1_EXPORT Autolabel_trainer
{
  public:
    std::string m_input_dir;
    std::string m_task;

  public:
    Autolabel_trainer () {}
    ~Autolabel_trainer () {}

  public:
    void load_input_dir (const char* input_dir);
    void set_task (const char* task);
    void save_libsvm (const char* output_libsvm_fn);

  private:
    void load_input_dir_recursive (std::string input_dir);
    void load_input_file (const char* nrrd_fn, const char* fcsv_fn);
    void load_input_file_la (const char* nrrd_fn, const char* fcsv_fn);
    void load_input_file_tsv1 (const char* nrrd_fn, const char* fcsv_fn);
    void load_input_file_tsv2 (const char* nrrd_fn, const char* fcsv_fn);
};

#endif
