/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include "plm_config.h"
#include "fdk_opts.h"
#include "volume.h"

int CUDA_reconstruct_conebeam (Volume *vol, MGHCBCT_Options *options);

int
main(int argc, char* argv[])
{
    MGHCBCT_Options options;
    Volume* vol;

    ////////////////////////////////////
#if defined (READ_PFM)
    printf("[PFM Input]\n");
#else
    printf("[PGM Input]\n"); 
#endif
    ////////////////////////////////////


    /**************************************************************** 
     * STEP 0: Parse commandline arguments                           * 
     ****************************************************************/
    parse_args (&options, argc, argv);
	
    /*****************************************************
     * STEP 1: Create the 3D array of voxels              *
     *****************************************************/
    vol = my_create_volume (&options);

    /***********************************************
     * STEP 2: Reconstruct/populate the volume      *
     ***********************************************/
    CUDA_reconstruct_conebeam (vol, &options);	

    /*************************************
     * STEP 3: Convert to HU values       *
     *************************************/
    convert_to_hu (vol, &options);

    /*************************************
     * STEP 4: Write MHA output file      *
     *************************************/
    printf("Writing output volume...");
    write_mha (options.output_file, vol);
    printf(" done.\n\n");

    return 0;
}
