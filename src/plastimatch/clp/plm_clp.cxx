/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include "plm_clp.h"

namespace dlib {

void 
Plm_clp::add_default_options (void)
{
    this->add_long_option ("h", "help", "display this help message");
    this->add_long_option ("", "version", "display the program version");
}

void 
Plm_clp::check_default_options (void)
{
    /* Check if the -h option was given */
    if (this->option("h") || this->option("help")) {
        if (this->number_of_arguments()) {
            /* Secret option.  If you use "--help something", then you 
               get a help with narrower format, which can be pasted 
               into sphinx documentation. */
            this->wrap_len = 73;
            usage_fn (this, argc, argv);
        } else {
            usage_fn (this, argc, argv);
        }
        exit (0);
    }

    if (this->option("version")) {
        std::cout << "Plastimatch version " << PLASTIMATCH_VERSION_STRING
            << std::endl;
        exit (0);
    }
}

void 
Plm_clp::add_long_option (
    const string_type& short_name,
    const string_type& long_name,
    const string_type& description,
    unsigned long number_of_arguments,
    const string_type& default_value)
{
    if (short_name == "" && long_name == "") return;

    std::string key;
    std::string option_val;
    std::string description_val;
    if (long_name == "") {
        /* Only short */
        key = short_name;
        option_val = "  -" + short_name;
        this->add_option_with_default (short_name, description, 
            number_of_arguments, default_value);
    } 
    else if (short_name == "") {
        /* Only long */
        key = long_name;
        option_val = "      --" + long_name;
        this->add_option_with_default (long_name, description, 
            number_of_arguments, default_value);
    }
    else {
        /* Both long and short */
        key = long_name;
        option_val = "  -" + short_name + ", --" + long_name;
        this->add_option_with_default (short_name, description, 
            number_of_arguments, "");
        this->add_option_with_default (long_name, description, 
            number_of_arguments, default_value);
        this->long_to_short_map.insert (
            std::pair<string_type,string_type> (long_name, short_name));
        this->short_to_long_map.insert (
            std::pair<string_type,string_type> (short_name, long_name));
    }

    option_map.insert (
        std::pair<string_type,string_type> (key, option_val));
    description_map.insert (
        std::pair<string_type,string_type> (key, description));
}

void 
Plm_clp::assign_plm_long_13 (
    plm_long *arr, 
    const string_type& name)
{
	int rc;
	unsigned int a, b, c;
	rc = sscanf (get_string(name).c_str(), "%d %d %d", &a, &b, &c);
	if (rc == 1) {
	    arr[0] = a;
	    arr[1] = a;
	    arr[2] = a;
	} else if (rc == 3) {
	    arr[0] = a;
	    arr[1] = b;
	    arr[2] = c;
	} else {
	    string_type error_string = 
		"Error. Option "
		+ get_option_string (name) 
		+ " takes one or three integer arguments.";
	    throw dlib::error (error_string);
	}
    }

void Plm_clp::assign_float_vec (
    std::vector<float>* float_vec, 
    const string_type& name)
{
    std::istringstream str (get_string (name));
    str.exceptions(std::ios::badbit | std::ios::failbit);
    try {
        float fv;
        while (str >> fv) {
            float_vec->push_back(fv);
        }
    }
    catch (std::ios_base::failure e) {
        if (str.fail() && !str.eof()) {
            string_type error_string = 
                "Error parsing option "
                + get_option_string (name) 
                + ", it contains non-float values.";
            throw dlib::error (error_string);
        }
    }
}

}  /* end namespace */
