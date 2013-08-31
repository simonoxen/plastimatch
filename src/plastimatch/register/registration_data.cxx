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

Registration_data::Registration_data ()
{
    fixed_image = 0;
    moving_image = 0;
    fixed_roi = 0;
    moving_roi = 0;
    fixed_landmarks = 0;
    moving_landmarks = 0;
}

Registration_data::~Registration_data ()
{
    if (fixed_landmarks) delete fixed_landmarks;
    if (moving_landmarks) delete moving_landmarks;
    if (fixed_roi) delete fixed_roi;
    if (moving_roi) delete moving_roi;
}

void
Registration_data::load_global_input_files (Registration_parms* regp)
{
    Plm_image_type image_type = PLM_IMG_TYPE_ITK_FLOAT;
    Shared_parms *shared = regp->get_shared_parms();

    /* Load images */
    logfile_printf ("Loading fixed image: %s\n", 
        regp->get_fixed_fn().c_str());
    this->fixed_image = plm_image_load (regp->get_fixed_fn(), image_type);

    logfile_printf ("Loading moving image: %s\n", 
        regp->get_moving_fn().c_str());
    this->moving_image = plm_image_load (regp->get_moving_fn(), image_type);

    /* load "global" rois */
    if (shared->fixed_roi_fn != "") {
        logfile_printf ("Loading fixed roi: %s\n", 
            shared->fixed_roi_fn.c_str());
        this->fixed_roi = plm_image_load (
            shared->fixed_roi_fn, PLM_IMG_TYPE_ITK_UCHAR);
    } else {
        this->fixed_roi = 0;
    }
    if (shared->moving_roi_fn != "") {
        logfile_printf ("Loading moving roi: %s\n", 
            shared->moving_roi_fn.c_str());
        this->moving_roi = plm_image_load (
            shared->moving_roi_fn, PLM_IMG_TYPE_ITK_UCHAR);
    } else {
        this->moving_roi = 0;
    }

    /* Load landmarks */
    if (regp->fixed_landmarks_fn.not_empty()) {
        if (regp->moving_landmarks_fn.not_empty()) {
            logfile_printf ("Loading fixed landmarks: %s\n", 
                (const char*) regp->fixed_landmarks_fn);
            fixed_landmarks = new Labeled_pointset;
            fixed_landmarks->load_fcsv (
                (const char*) regp->fixed_landmarks_fn);
            logfile_printf ("Loading moving landmarks: %s\n", 
                (const char*) regp->moving_landmarks_fn);
            moving_landmarks = new Labeled_pointset;
            moving_landmarks->load_fcsv (
                (const char*) regp->moving_landmarks_fn);
        } else {
            print_and_exit (
                "Sorry, you need to specify both fixed and moving landmarks");
        }
    }
    else if (regp->moving_landmarks_fn.not_empty()) {
        print_and_exit (
            "Sorry, you need to specify both fixed and moving landmarks");
    }
    else if (regp->fixed_landmarks_list.not_empty()) {
        if (regp->moving_landmarks_list.not_empty()) {
            fixed_landmarks = new Labeled_pointset;
            moving_landmarks = new Labeled_pointset;
            fixed_landmarks->set_ras (regp->fixed_landmarks_list);
            moving_landmarks->set_ras (regp->moving_landmarks_list);
        }
    }
}

void
Registration_data::load_stage_input_files (Stage_parms* stage)
{
    Shared_parms *shared = stage->get_shared_parms();

    if (shared->fixed_roi_fn != "") {
        logfile_printf ("Loading fixed roi: %s\n", 
            shared->fixed_roi_fn.c_str());
        this->fixed_roi = plm_image_load (
            shared->fixed_roi_fn, PLM_IMG_TYPE_ITK_UCHAR);
    }

    if (shared->moving_roi_fn != "") {
        logfile_printf ("Loading moving roi: %s\n", 
            shared->moving_roi_fn.c_str());
        this->moving_roi = plm_image_load (
            shared->moving_roi_fn, PLM_IMG_TYPE_ITK_UCHAR);
    }
}
