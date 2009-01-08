#include <stdio.h>
#include <iostream>
#include <vector>
#include "plastimatch-slicerCLP.h"
#include "plm_registration.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDiscreteGaussianImageFilter.h"


int 
main (int argc, char * argv [])
{
    PARSE_ARGS;

    char buf1[L_tmpnam+1];
    char* parms_fn = tmpnam (buf1);
    //char* parms_fn = "C:/tmp/plastimatch-slicer-parms.txt";
    FILE* fp = fopen (parms_fn, "w");

    fprintf (fp,
	     "[GLOBAL]\n"
	     "fixed=%s\n"
	     "moving=%s\n"
//	     "xf_out=%s\n"
//	     "vf_out=%s\n"
	     "img_out=%s\n\n"
	     "[STAGE]\n"
	     "metric=%s\n"
	     "xform=bspline\n"	
	     "optim=lbfgsb\n"
	     "impl=gpuit_cpu\n"
	     "max_its=%d\n"
	     "convergence_tol=5\n"
	     "grad_tol=1.5\n"
	     "res=%d %d %d\n"
	     "grid_spac=%g %g %g\n",
	     /* Global */
	     plmslc_fixed_volume.c_str(),
	     plmslc_moving_volume.c_str(),
//	     "C:/tmp/plmslc-xf.txt",
//	     "C:/tmp/plmslc-vf.mha",
	     plmslc_warped_volume.c_str(),
	     /* Stage 1 */
	     metric.c_str(),
	     stage_1_its,
	     stage_1_resolution[0],
	     stage_1_resolution[1],
	     stage_1_resolution[2],
	     stage_1_grid_size,
	     stage_1_grid_size,
	     stage_1_grid_size
	     );
    if (enable_stage_2) {
	fprintf (fp, 
	     "[STAGE]\n"
	     "xform=bspline\n"
	     "optim=lbfgsb\n"
	     "impl=gpuit_cpu\n"
	     "max_its=%d\n"
	     "convergence_tol=5\n"
	     "grad_tol=1.5\n"
	     "res=%d %d %d\n"
	     "grid_spac=%g %g %g\n",
	     /* Stage 2 */
	     stage_2_its,
	     stage_2_resolution[0],
	     stage_2_resolution[1],
	     stage_2_resolution[2],
	     stage_2_grid_size,
	     stage_2_grid_size,
	     stage_2_grid_size
	     );
    }
    fclose (fp);

#if defined (commentout)
    fp = fopen (ff_fn, "w");
    //fprintf (fp, "Fixed fiducials:\n");
    for (int i = 0; i < plmslc_fixed_fiducials.size(); i++) {
      fprintf (fp, "%g %g %g\n", 
	       plmslc_fixed_fiducials[i][0],
	       plmslc_fixed_fiducials[i][1],
	       plmslc_fixed_fiducials[i][2]
	       );
    }
    fclose (fp);

    fp = fopen (mf_fn, "w");
    //fprintf (fp, "Moving fiducials:\n");
    for (int i = 0; i < plmslc_moving_fiducials.size(); i++) {
      fprintf (fp, "%g %g %g\n", 
	       plmslc_moving_fiducials[i][0],
	       plmslc_moving_fiducials[i][1],
	       plmslc_moving_fiducials[i][2]
	       );
    }
    fclose (fp);
#endif

    Registration_Parms regp;
    if (parse_command_file (&regp, parms_fn) < 0) {
	return EXIT_FAILURE;
    }
    do_registration (&regp);
    return EXIT_SUCCESS;
}
