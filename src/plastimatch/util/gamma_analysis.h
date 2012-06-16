/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/*! \file gamma_analysis.h
    \brief Documented file for gamma analysis.
*   This is a method for comparison two dose distributions. 
*   It simultaneously considers the dose difference and distance-to-agreement. 
*/

#ifndef _gamma_analysis_h_
#define _gamma_analysis_h_

#include "plmutil_config.h"
#include "direction_cosines.h"
#include "plm_macros.h"
#include "itk_image_type.h"

class Plm_image;

/*! \enum Gamma_output_mode Selector for output image type (gamma, or binary pass/fail)
*/
enum Gamma_labelmap_mode {
    NONE,  /*!< disable output binary uchar labelmap for gamma */
    PASS,  /*!< output binary (1/0) image of type uchar, 1 if gamma<1 */ 
    FAIL   /*!< output binary (1/0) image of type uchar, 1 if gamma>1 */ 
};

/*! \class Gamma_parms
    \brief This is the Gamma_parms class.
    * Used to pass input and output parameters for gamma analysis
	to/from find_dose_threshold() and do_gamma_analysis() */
class Gamma_parms {
public:
    
    float r_tol, /*!< distance-to-agreement (DTA) criterion, input parameter */ 
        d_tol, /*!< dose-difference criterion, input for do_gamma_analysis, in Gy 
                 Set from 3D Slicer plugin either directly or as percentrage of
                 dose_max, as found by find_dose_threshold().
               */
        dose_max, /*!< maximum dose (max voxel value) in the img_in1, set by find_dose_threshold() */
        gamma_max; /*!< maximum gamma to calculate */

    Direction_cosines dc; /*!< orientation of the output image (currently unused, set to identity?) */

    Plm_image *img_in1; /*!< input dose image 1 for gamma analysis*/
    Plm_image *img_in2; /*!< input dose image 2 for gamma analysis*/
    Plm_image *img_out; /*!< output float type image, voxel value = calculated gamma value */
    Plm_image *labelmap_out; /*!< output uchar type labelmap, voxel value = 1/0 for pass/fail */

    Gamma_labelmap_mode mode; /*!< output mode selector for 3D Slicer plugin*/

public:
    Gamma_parms () { /*!< Constructor for Gamma_parms, sets default values (mode GAMMA) 
                       for the 3D Slicer plugin */
        img_in1 = 0;
        img_in2 = 0;
        img_out = 0;
        labelmap_out = 0;
        r_tol = d_tol = gamma_max = 3; 
        mode = NONE;
    }
};

/*! \fn PLMUTIL_C_API void find_dose_threshold (Gamma_parms *parms);
    \brief This function finds a dose threshold.  It is a global function. 
	parms->dose_max is output = maximum voxel value within parms->img_in_1
*/
/*! \fn PLMUTIL_C_API void do_gamma_analysis (Gamma_parms *parms);
    \brief This function calculates gamma index.  It is a global function. 
	parms->img_out is output
*/

PLMUTIL_C_API void find_dose_threshold (Gamma_parms *parms);
PLMUTIL_C_API void do_gamma_analysis (Gamma_parms *parms);
PLMUTIL_C_API void do_gamma_threshold (Gamma_parms *parms);

#endif
