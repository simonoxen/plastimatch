/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "plastimatch_slicer_dose_comparisonCLP.h"

#include "plm_image.h"
#include "plm_file_format.h"
#include "itk_image.h"
#include "itk_image_save.h"
#include "pstring.h"
#include <string.h>
#include <stdlib.h>
#include "pcmd_diff.h"



int
main (int argc, char * argv [])
{
	PARSE_ARGS;
	
	Diff_parms parms;
	
	bool have_dose1_img_input = false;
	bool have_dose2_img_input = false;
    bool have_dose_diff_output = false;

    /* Input dose */
    if (plmslc_dose1_input_img != "" 
	&& plmslc_dose1_input_img != "None")
    {
	have_dose1_img_input = true;
	parms.img_in_1_fn = plmslc_dose1_input_img.c_str();
    }

    if (plmslc_dose2_input_img != "" 
	&& plmslc_dose2_input_img != "None")
    {
	have_dose2_img_input = true;
	parms.img_in_2_fn = plmslc_dose2_input_img.c_str();
    }

    if (!have_dose1_img_input && !have_dose2_img_input) {
	printf ("Error.  No input specified.\n");
	return EXIT_FAILURE;
    }

    /* Output dose difference */
    if (plmslc_dose_diff_output_img != "" 
	&& plmslc_dose_diff_output_img != "None")
    {
	have_dose_diff_output = true;
	parms.img_out_fn = plmslc_dose_diff_output_img.c_str();
    }

    
	/* Output type 
    if (plmslc_xformwarp_output_type != "auto") {
	parms.output_type = plm_image_type_parse (
	    plmslc_xformwarp_output_type.c_str());
    }
	*/

    /* What is the input file type? 
    file_type = plm_file_format_deduce ((const char*) parms.input_fn);
	*/

    /* Process diff */

	diff_main (&parms);

    return EXIT_SUCCESS;
}
