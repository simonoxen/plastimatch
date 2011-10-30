/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _autolabel_trainer_h_
#define _autolabel_trainer_h_

#include "plm_config.h"
#include <map>
#include <stdio.h>
#include "pstring.h"

class Dlib_trainer;

class plastimatch1_EXPORT Autolabel_trainer
{
  public:
    Autolabel_trainer ();
    ~Autolabel_trainer ();

  public:
    std::string m_input_dir;
    std::string m_task;

  private:
    Dlib_trainer *m_dt;

  public:
    void save_csv (const char* output_csv_fn);
    void save_tsacc (const Pstring& output_tsacc_fn);
    void set_input_dir (const char* input_dir);
    void set_task (const char* task);
    void train (const Pstring& output_net_fn);

  private:
    void load_inputs ();
    void load_input_dir_recursive (std::string input_dir);
    void load_input_file (const char* nrrd_fn, const char* fcsv_fn);
    void load_input_file_la (const char* nrrd_fn, const char* fcsv_fn);
    void load_input_file_tsv1 (const char* nrrd_fn, const char* fcsv_fn);
    void load_input_file_tsv2 (const char* nrrd_fn, const char* fcsv_fn);
};

#endif
