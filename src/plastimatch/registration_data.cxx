/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "pointset.h"
#include "itk_image_load.h"
#include "logfile.h"
#include "registration_data.h"
#include "plm_parms.h"

Registration_data::Registration_data ()
{
    fixed_image = 0;
    moving_image = 0;
    fixed_mask = 0;
    moving_mask = 0;
    fixed_landmarks = 0;
    moving_landmarks = 0;
}

Registration_data::~Registration_data ()
{
    if (fixed_landmarks) delete fixed_landmarks;
    if (moving_landmarks) delete moving_landmarks;
}

void
Registration_data::load_input_files (Registration_parms* regp)
{
    Plm_image_type image_type = PLM_IMG_TYPE_ITK_FLOAT;

    /* Load images */
    logfile_printf ("Loading fixed image: %s\n", regp->fixed_fn);
    this->fixed_image = plm_image_load (regp->fixed_fn, image_type);

    logfile_printf ("Loading moving image: %s\n", regp->moving_fn);
    this->moving_image = plm_image_load (regp->moving_fn, image_type);

    /* Load masks */
    if (regp->fixed_mask_fn[0]) {
        logfile_printf ("Loading fixed mask: %s\n", regp->fixed_mask_fn);
//      this->fixed_mask = plm_image_load (regp->fixed_mask_fn, PLM_IMG_TYPE_ITK_FLOAT);
        this->fixed_mask = plm_image_load (regp->fixed_mask_fn, PLM_IMG_TYPE_ITK_UCHAR);
    } else {
        this->fixed_mask = 0;
    }
    if (regp->moving_mask_fn[0]) {
        logfile_printf ("Loading moving mask: %s\n", regp->moving_mask_fn);
//      this->moving_mask = plm_image_load (regp->moving_mask_fn, PLM_IMG_TYPE_ITK_FLOAT);
        this->moving_mask = plm_image_load (regp->moving_mask_fn, PLM_IMG_TYPE_ITK_UCHAR);
    } else {
        this->moving_mask = 0;
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
    } else if (regp->moving_landmarks_fn.not_empty()) {
        print_and_exit (
            "Sorry, you need to specify both fixed and moving landmarks");
    }
}
