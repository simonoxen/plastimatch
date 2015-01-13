/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ml_convert_h_
#define _ml_convert_h_

#include "plmsegment_config.h"
#include <string>

class Ml_convert_private;

class PLMSEGMENT_API Ml_convert
{
public:
    Ml_convert_private *d_ptr;
public:
    Ml_convert ();
    ~Ml_convert ();

public:
    void set_label_filename (const std::string& label_filename);
    void set_feature_directory (const std::string& feature_dir);
    void set_output_filename (const std::string& output_filename);
    void run ();
};


#endif
