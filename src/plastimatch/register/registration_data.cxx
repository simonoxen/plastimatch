/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"

#include "logfile.h"
#include "plm_image.h"
#include "plm_image_type.h"
#include "print_and_exit.h"
#include "registration_data.h"
#include "registration_parms.h"
#include "shared_parms.h"
#include "stage_parms.h"

#define DEFAULT_IMAGE_KEY "0"

class Registration_data_private
{
public:
    Stage_parms auto_parms;
public:
    Registration_data_private () {
    }
    ~Registration_data_private () {
    }
};

Registration_data::Registration_data ()
{
    fixed_landmarks = 0;
    moving_landmarks = 0;
    d_ptr = new Registration_data_private;
}

Registration_data::~Registration_data ()
{
    if (fixed_landmarks) delete fixed_landmarks;
    if (moving_landmarks) delete moving_landmarks;
    delete d_ptr;
}

void
Registration_data::load_global_input_files (Registration_parms::Pointer& regp)
{
    this->load_shared_input_files (regp->get_shared_parms());
}

void
Registration_data::load_stage_input_files (const Stage_parms* stage)
{
    this->load_shared_input_files (stage->get_shared_parms());
}

void
Registration_data::load_shared_input_files (const Shared_parms* shared)
{
    /* Load images */
    std::map<std::string,std::string>::const_iterator fix_it;
    for (fix_it = shared->fixed_fn.begin();
         fix_it != shared->fixed_fn.end(); ++fix_it)
    {
        logfile_printf ("Loading fixed image [%s]: %s\n",
            fix_it->first.c_str(), fix_it->second.c_str());
        this->set_fixed_image (fix_it->first,
            Plm_image::New (fix_it->second, PLM_IMG_TYPE_ITK_FLOAT));
        break;
    }
    std::map<std::string,std::string>::const_iterator mov_it;
    for (mov_it = shared->moving_fn.begin();
         mov_it != shared->moving_fn.end(); ++mov_it)
    {
        logfile_printf ("Loading moving image [%s]: %s\n",
            mov_it->first.c_str(), mov_it->second.c_str());
        this->set_moving_image (mov_it->first, 
            Plm_image::New (mov_it->second, PLM_IMG_TYPE_ITK_FLOAT));
    }

    /* load "global" rois */
    if (shared->fixed_roi_fn != "") {
        logfile_printf ("Loading fixed roi: %s\n", 
            shared->fixed_roi_fn.c_str());
        this->fixed_roi = Plm_image::New (
            shared->fixed_roi_fn, PLM_IMG_TYPE_ITK_UCHAR);
    }
    if (shared->moving_roi_fn != "") {
        logfile_printf ("Loading moving roi: %s\n", 
            shared->moving_roi_fn.c_str());
        this->moving_roi = Plm_image::New (
            shared->moving_roi_fn, PLM_IMG_TYPE_ITK_UCHAR);
    }

    /* load stiffness */
    if (shared->fixed_stiffness_fn != "") {
        logfile_printf ("Loading fixed stiffness: %s\n", 
            shared->fixed_stiffness_fn.c_str());
        this->fixed_stiffness = Plm_image::New (
            shared->fixed_stiffness_fn, PLM_IMG_TYPE_ITK_FLOAT);
    }

    /* load landmarks */
    if (shared->fixed_landmarks_fn != "") {
        if (shared->moving_landmarks_fn != "") {
            logfile_printf ("Loading fixed landmarks: %s\n", 
                shared->fixed_landmarks_fn.c_str());
            fixed_landmarks = new Labeled_pointset;
            fixed_landmarks->load_fcsv (
                shared->fixed_landmarks_fn.c_str());
            logfile_printf ("Loading moving landmarks: %s\n", 
                shared->moving_landmarks_fn.c_str());
            moving_landmarks = new Labeled_pointset;
            moving_landmarks->load_fcsv (
                shared->moving_landmarks_fn.c_str());
        } else {
            print_and_exit (
                "Sorry, you need to specify both fixed and moving landmarks");
        }
    }
    else if (shared->moving_landmarks_fn != "") {
        print_and_exit (
            "Sorry, you need to specify both fixed and moving landmarks");
    }
    else if (shared->fixed_landmarks_list != ""
        && shared->moving_landmarks_list != "")
    {
        fixed_landmarks = new Labeled_pointset;
        moving_landmarks = new Labeled_pointset;
        fixed_landmarks->insert_ras (shared->fixed_landmarks_list.c_str());
        moving_landmarks->insert_ras (shared->moving_landmarks_list.c_str());
    }
}

void
Registration_data::set_fixed_image (const Plm_image::Pointer& image)
{
    this->fixed_image[DEFAULT_IMAGE_KEY] = image;
}

void
Registration_data::set_fixed_image (
    const std::string& index,
    const Plm_image::Pointer& image)
{
    if (index == "") {
        this->fixed_image[DEFAULT_IMAGE_KEY] = image;
    } else {
        this->fixed_image[index] = image;
    }
}

void
Registration_data::set_moving_image (const Plm_image::Pointer& image)
{
    this->set_moving_image (DEFAULT_IMAGE_KEY, image);
}

void
Registration_data::set_moving_image (
    const std::string& index,
    const Plm_image::Pointer& image)
{
    if (index == "") {
        this->moving_image[DEFAULT_IMAGE_KEY] = image;
    } else {
        this->moving_image[index] = image;
    }
}

Plm_image::Pointer&
Registration_data::default_fixed_image ()
{
    return this->fixed_image[DEFAULT_IMAGE_KEY];
}

Plm_image::Pointer&
Registration_data::default_moving_image ()
{
    return this->moving_image[DEFAULT_IMAGE_KEY];
}

Stage_parms*
Registration_data::get_auto_parms ()
{
    return &d_ptr->auto_parms;
}
