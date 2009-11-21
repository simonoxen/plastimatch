#include "plm_config.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include "plastimatch-slicer-tpsCLP.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "tps.h"

int 
main (int argc, char * argv [])
{
    PARSE_ARGS;

    //FILE* fp = tmpfile ();

    //char* fn = "C:/tmp/plastimatch-slicer-parms.txt";
    char* fn = "/tmp/plastimatch-slicer-tps.txt";
    FILE* fp = fopen (fn, "w");

    unsigned long num_fiducials = plmslc_fixed_fiducials.size();
    if (plmslc_moving_fiducials.size() < num_fiducials) {
	num_fiducials = plmslc_moving_fiducials.size();
    }

    /* GCS FIX: This is a waste to load the whole image just to get 
       the header. */
    PlmImage *pli_fixed = plm_image_load_native (plmslc_fixed_volume.c_str());
    PlmImageHeader plih;
    plih.set_from_plm_image (pli_fixed);

    fprintf (fp, "PLASTIMATCH_TPS_XFORM <experimental>\n");
    fprintf (fp, "img_origin = %g %g %g\n",
	plih.m_origin[0], plih.m_origin[1], plih.m_origin[2]);
    fprintf (fp, "img_spacing = %g %g %g\n",
	plih.m_spacing[0], plih.m_spacing[1], plih.m_spacing[2]);
    fprintf (fp, "img_dim = %d %d %d\n",
	plih.Size(0), plih.Size(1), plih.Size(2));

    for (unsigned long i = 0; i < num_fiducials; i++) {
	float src[3], tgt[3];

	for (int d = 0; d < 3; d++) {
	    src[d] = plmslc_fixed_fiducials[i][d];
	    tgt[d] = plmslc_moving_fiducials[i][d];
	}
	float alpha = tps_default_alpha (src, tgt);

	/* Only RAS coordinates seem to work in slicer.  Change to LPS */
	fprintf (fp, "%g %g %g %g %g %g %g\n", 
	    -plmslc_fixed_fiducials[i][0],
	    -plmslc_fixed_fiducials[i][1],
	    plmslc_fixed_fiducials[i][2],
	    -plmslc_moving_fiducials[i][0],
	    -plmslc_moving_fiducials[i][1],
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
