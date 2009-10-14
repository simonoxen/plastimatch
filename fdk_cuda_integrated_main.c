/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */

//#define _CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>


#include <stdio.h>
#include <stdlib.h>
#include "plm_config.h"
#include "fdk_opts_ext.h"
#include "fdk_utils_ext.h"
#include "volume.h"
//#include "volume_ext.h"
#include "readmha_ext.h"
#include "MGHDRR_Options.h"
#include "MGHMtx_opts.h"
#include "write_matrix.h"
#include "file_util.h"
#if (_WIN32)
#include <windows.h>
#endif

//extern "C"
//{
int CUDA_reconstruct_conebeam (Volume *vol, MGHCBCT_Options_ext *options);
int CUDA_DRR (Volume *vol, MGHCBCT_Options_ext *options);
//}
int
main(int argc, char* argv[]) 
{
    //memoryleak test. No leakage dected
    //_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
    int i;
    MGHCBCT_Options_ext options;
    MGHMtx_Options matrix_options;

    Volume* vol;
    char matrix_ProjAngle_file [256];
    char matrix_output_prefix [256];
    char dirbuf[256];
    char argbuf[256];
    int CheckNormExits;
	

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

    parse_args_ext (&options, argc, argv);
    wm_set_default_options (&matrix_options);
    //give the previously allocated array to pointers.
	
    strcpy(argbuf,argv[0]);
    for(i=strlen(argbuf)-1;i>=0;i--)
	if (argbuf[i]=='\\'||argbuf[i]=='/'){
	    argbuf[i+1]='\0';
	    break;
	}
    if (options.full_fan){
	strcat(argbuf,options.Full_normCBCT_name);
	options.Full_normCBCT_name=argbuf;
    }
    else{
	strcat(argbuf,options.Half_normCBCT_name);
	options.Half_normCBCT_name=argbuf;
    }
	

    strcpy(matrix_ProjAngle_file ,options.input_dir);
    strcat(matrix_ProjAngle_file, "\\ProjAngles.txt");
    matrix_options.ProjAngle_file=matrix_ProjAngle_file;
	

    strcpy(dirbuf,options.input_dir);
    CreateDirectory(strcat(dirbuf,"\\tmp"),NULL);
    //make_directory (strcat (dirbuf, "/tmp"));

    strcpy(matrix_output_prefix ,options.input_dir);
    strcat(matrix_output_prefix , "\\tmp\\out_");
    matrix_options.output_prefix=matrix_output_prefix;
	

    if (!options.full_fan){
	matrix_options.image_resolution[0]=384;
	matrix_options.image_resolution[1]=512*2;
	matrix_options.image_size[0]=300;
	matrix_options.image_size[1]=400*2;
    }
    else{
	matrix_options.image_resolution[0]=384;
	matrix_options.image_resolution[1]=512;
	matrix_options.image_size[0]=300;
	matrix_options.image_size[1]=400;
    }
    write_matrix(&matrix_options);
    printf("Write matrix OK");
    fflush(stdout);
	

#if 1 
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
    if (options.full_fan)
	CheckNormExits = file_exists (options.Full_normCBCT_name);
    else
	CheckNormExits = file_exists (options.Half_normCBCT_name);
    if (CheckNormExits) {
	bowtie_correction(vol,&options);
    }
    else{
	printf("%s\n%s\n",options.Full_normCBCT_name,options.Half_normCBCT_name);
	printf("%s\n",argv[0]);
	printf("Skip bowtie correction because norm files do not exits\n");
    }


    /*************************************
     * STEP 4: Write MHA output file      *
     *************************************/
    printf("Writing output volume...");	
    write_mha_512prefix (options.output_file, vol,&options);
    printf(" done.\n\n");
    free(vol->img);
    free(vol);

#else
	vol=read_mha_512prefix(options.output_file);
	CUDA_DRR(vol, &options);
	free(vol->img);
	free(vol);
#endif
    return 0;
}
