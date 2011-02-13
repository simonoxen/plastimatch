/* -----------------------------------------------------------------------
   This file is derived from dlib source code, http://dlib.net, and 
   is licensed under the "Boost Software License - Version 1.0".
   Modifications by Greg Sharp <gregsharp@geocities.com>
   ----------------------------------------------------------------------- */
#ifndef _plm_dlib_clp_h_
#define _plm_dlib_clp_h_

#include "plm_config.h"
#include <stdio.h>
#include <iostream>
#include <map>
#include <vector>

#include "dlib/cmd_line_parser.h"

typedef dlib::cmd_line_parser<char>::check_1a_c Clp;

namespace dlib {
class Plm_clp
    : public cmd_line_parser<char>::check_1a_c 
{
public:
    std::map<string_type,string_type> default_value_map;
    std::map<string_type,string_type> option_map;
    std::map<string_type,string_type> description_map;
public:
    void 
    add_long_option (
	const string_type& short_name,
	const string_type& long_name,
	const string_type& description,
	unsigned long number_of_arguments = 0,
	const string_type& default_value = ""
    ) {
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
		number_of_arguments, default_value);
	    this->add_option_with_default (long_name, description, 
		number_of_arguments, default_value);
	}

	option_map.insert (
	    std::pair<string_type,string_type> (key, option_val));
	description_map.insert (
	    std::pair<string_type,string_type> (key, description));
    }

    void 
    add_option_with_default (
	const string_type& name,
	const string_type& description,
	unsigned long number_of_arguments = 0,
	const string_type& default_value = ""
    ) {
	this->add_option (name, description, number_of_arguments);
	if (default_value != "") {
	    default_value_map.insert (
		std::pair<string_type,string_type> (name, default_value));
	}
    }

    template <class T>
    void
    get_value (
	T& dest,
	const string_type& name
    ) {
	try {
	    dest = dlib::sa = this->get_value (name);
	}
	catch (std::exception& e) {
	    string_type error_string = "Error. Option --" 
		+ name + " had an illegal or missing argument.";
	    throw dlib::error (error_string);
	}
    }

    string_type 
    get_value (
	const string_type& name
    ) {
	/* Option specified on command line */
	if (this->option(name)) {
	    return this->option(name).argument();
	}

	/* Default value */
	std::map<string_type,string_type>::iterator it;
	it = this->default_value_map.find (name);
	if (it != this->default_value_map.end()) {
	    return it->second;
	}

	/* Not specified on command line, and no default value */
	return "";
    }

    /* Shorthand functions for specific well-known types */
    void assign_int13 (int *arr, const string_type& name) {
	int rc;
	rc = sscanf (get_cstring (name), "%d %d %d", 
	    &arr[0], &arr[1], &arr[2]);
	if (rc == 1) {
	    arr[1] = arr[2] = arr[0];
	} else if (rc != 3) {
	    string_type error_string = "Error. Option --" 
		+ name + " takes one or three integer arguments.";
	    throw dlib::error (error_string);
	}
    }
    void assign_float13 (float *arr, const string_type& name) {
	float rc;
	rc = sscanf (get_cstring (name), "%g %g %g", 
	    &arr[0], &arr[1], &arr[2]);
	if (rc == 1) {
	    arr[1] = arr[2] = arr[0];
	} else if (rc != 3) {
	    string_type error_string = "Error. Option --" 
		+ name + " takes one or three float arguments.";
	    throw dlib::error (error_string);
	}
    }
    const char* get_cstring (const string_type& name) {
	return get_value (name).c_str();
    }
    float get_float (const string_type& name) {
	float out;
	get_value (out, name);
	return out;
    }
    std::string get_string (const string_type& name) {
	return get_value (name);
    }

    void 
    print_options (
        std::basic_ostream<char>& out
    ) {
        typedef char ct;
        typedef std::basic_string<ct> string;
        typedef string::size_type size_type;

        try {
            out << _dT(ct,"Options:");

            // this loop here is just the bottom loop but without the print 
	    // statements.  I'm doing this to figure out what len should be.
            size_type max_len = 0; 
            this->reset();
            while (this->move_next()) 
	    {
		/* Skip past options which aren't in the map */
		const std::string name = this->element().name();
		std::map<string_type,string_type>::iterator it;
		it = this->option_map.find (name);
		if (it == this->option_map.end()) {
		    continue;
		}

                size_type len = 0; 
		len = it->second.size();

                if (this->element().number_of_arguments() == 1) {
                    len += 6;
                } else {
                    for (unsigned long i = 0; 
			 i < this->element().number_of_arguments(); ++i)
                    {
                        len += 7;
                        if (i+1 > 9)
                            ++len;
                    }
                }

                len += 3;
                if (len < 33)
                    max_len = std::max(max_len,len);
            }

            this->reset();
            while (this->move_next())
            {
		/* Skip past options which aren't in the map */
		const std::string name = this->element().name();
		std::map<string_type,string_type>::iterator it;
		it = this->option_map.find (name);
		if (it == this->option_map.end()) {
		    continue;
		}

                size_type len = 0; 
		out << _dT(ct,"\n");
		out << it->second;
		len = it->second.size();

                if (this->element().number_of_arguments() == 1) {
                    out << _dT(ct," <arg>");
                    len += 6;
                } else {
                    for (unsigned long i = 0; 
			 i < this->element().number_of_arguments(); ++i)
                    {
                        out << _dT(ct," <arg") << i+1 << _dT(ct,">");
                        len += 7;
                        if (i+1 > 9)
                            ++len;
                    }
                }

                out << "   ";
                len += 3;

                while (len < max_len) {
                    ++len;
                    out << " ";
                }

                const unsigned long ml = static_cast<unsigned long>(max_len);
                // now print the description but make it wrap around 
		// nicely if it is to long to fit on one line.
                if (len <= max_len)
                    out << wrap_string(this->element().description(),0,ml+1);
                else
                    out << "\n" 
			<< wrap_string(this->element().description(),ml,ml+1);
            }
            this->reset();
        }
        catch (...)
        {
            this->reset();
            throw;
        }
    }

}; /* end class */
}  /* end namespace */

#endif
