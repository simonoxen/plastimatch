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
    void set_append_filename (const std::string& append_filename);
    void set_label_filename (const std::string& label_filename);
    void set_output_filename (const std::string& output_filename);
    void set_output_format (const std::string& output_format);
    void add_feature_path (const std::string& feature_path);
    void run ();
};


#endif
