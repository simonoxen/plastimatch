#include <stdio.h>
#include <iostream>
#include <vector>
#include "dicomrt-import-slicerCLP.h"

#include "plm_config.h"
#include "warp_main.h"

int 
main (int argc, char * argv [])
{
    PARSE_ARGS;

    Warp_parms parms;
    char buf1[L_tmpnam+1];
    //    char* parms_fn = tmpnam (buf1);
    char* parms_fn = "C:/tmp/dicomrt-import-slicer-parms.txt";
    FILE* fp = fopen (parms_fn, "w");

    fprintf (fp,
	     "structure_set = %s\n"
	     "reference_vol = %s\n"
	     "output_labelmap = %s\n\n",
	     input_dicomrt_ss.c_str(),
	     reference_vol.c_str(),
	     output_labelmap.c_str()
	     );

    fclose (fp);

    warp_dicom_rtss (&parms);

    return EXIT_SUCCESS;
}
