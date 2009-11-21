#include <stdio.h>
#include <iostream>
#include <vector>
#include "plastimatch-slicer-tpsCLP.h"
#include "plm_registration.h"

int 
main (int argc, char * argv [])
{
    PARSE_ARGS;

    //FILE* fp = tmpfile ();

    //char* fn = "C:/tmp/plastimatch-slicer-parms.txt";
    char* fn = "/tmp/plastimatch-slicer-tps.txt";
    FILE* fp = fopen (fn, "w");

    fprintf (fp, "PLASTIMATCH_TPS_XFORM <experimental>\n");

    int num_fiducials = plmslc_fixed_fiducials.size();
    if (plmslc_moving_fiducials.size() < num_fiducials) {
	num_fiducials = plmslc_moving_fiducials.size();
    }

    for (int i = 0; i < num_fiducials; i++) {
	float alpha = tps_default_alpha (
	    plmslc_fixed_fiducials[i],
	    plmslc_moving_fiducials[i]);

	fprintf (fp, "%g %g %g %g %g %g %g\n", 
	    plmslc_fixed_fiducials[i][0],
	    plmslc_fixed_fiducials[i][1],
	    plmslc_fixed_fiducials[i][2]
	    plmslc_moving_fiducials[i][0],
	    plmslc_moving_fiducials[i][1],
	    plmslc_moving_fiducials[i][2],
	    alpha
	);
    }
    fclose (fp);

#if defined (commentout)
    Registration_Parms regp;
    if (parse_command_file (&regp, parms_fn) < 0) {
	return EXIT_FAILURE;
    }
    do_registration (&regp);
#endif

    return EXIT_SUCCESS;
}
