/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _registration_h_
#define _registration_h_

#include "plmregister_config.h"
#include "registration_data.h"
#include "registration_parms.h"
#include "xform.h"

class Registration_data;
class Registration_parms;
class Registration_private;

class PLMREGISTER_API Registration {
public:
    SMART_POINTER_SUPPORT (Registration);
    Registration_private *d_ptr;
public:
    Registration ();
    ~Registration ();
public:
    int set_command_file (const std::string& command_file);
    int set_command_string (const std::string& command_string);
    void set_fixed_image (Plm_image::Pointer& fixed);
    void set_moving_image (Plm_image::Pointer& moving);

    Registration_data::Pointer get_registration_data ();
    Registration_parms::Pointer get_registration_parms ();

    void do_registration ();
    Xform::Pointer do_registration_pure ();

};

#if defined (commentout)
PLMREGISTER_API Xform::Pointer do_registration_pure (
    Registration_data* regd,
    Registration_parms* regp
);
PLMREGISTER_API void do_registration (Registration_parms* regp);
#endif

#endif
