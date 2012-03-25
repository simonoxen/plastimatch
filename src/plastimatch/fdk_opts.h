/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_opts_h_
#define _fdk_opts_h_

#include "plm_config.h"
#include "plm_int.h"
#include "threading.h"

enum Fdk_filter_type {
    FDK_FILTER_TYPE_NONE,
    FDK_FILTER_TYPE_RAMP
};

#define OPTION_RESOLUTION_STRING "resolution"
#define OPTION_RESOLUTION 'r'
#define OPTION_INPUT_DIR_STRING "input-dir"
#define OPTION_INPUT_DIR 'I'
#define OPTION_OUTPUT_FILE_STRING "output-file"
#define OPTION_OUTPUT_FILE 'O'
#define OPTION_IMAGE_RANGE_STRING "image-range"
#define OPTION_IMAGE_RANGE 'a'
#define OPTION_SCALE_STRING "scale"
#define OPTION_SCALE 's'
#define OPTION_VOL_SIZE_STRING "volume-size"
#define OPTION_VOL_SIZE 'z'

typedef struct fdk_options Fdk_options;
struct fdk_options {
    Threading threading;

    int image_range_requested;
    int first_img;
    int skip_img;
    int last_img;

    plm_long dim[3];
    float vol_size[3];
    float xy_offset[2];

    float scale;
    enum Fdk_filter_type filter;

    char flavor;

    char* input_dir;
    char* output_file;

    int full_fan;            //Full_fan=1, Half_fan=0;
    char* Full_normCBCT_name;
    int Full_radius;
    char* Half_normCBCT_name;
    int Half_radius;
};

#if defined __cplusplus
extern "C" {
#endif

void fdk_parse_args (Fdk_options* options, int argc, char* argv[]);

#if defined __cplusplus
}
#endif

#endif
