/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _parameter_parser_h_
#define _parameter_parser_h_

#include "plmbase_config.h"

class PLMBASE_API Parameter_parser {
public:
    Parameter_parser ();
public:
    bool key_regularization;
public:
    /* Callbacks */
    virtual int process_section (
        const std::string& section) = 0;
    virtual int process_key_value (
        const std::string& section, 
        const std::string& key, 
        const std::string& val) = 0;

    /* Pass in "true" to enable key regularization, or "false" to 
       disable it.   Default is "true". */
    void enable_key_regularization (
        bool enable
    );

    /* Return zero if config string is correctly parsed */
    int parse_config_string (
        const char* config_string
    );
    int parse_config_string (
        const std::string& config_string
    );
    int parse_config_file (
        const char* config_fn
    );
    int parse_config_file (
        const std::string& config_fn
    );
};

#endif
