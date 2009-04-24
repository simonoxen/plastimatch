/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_opts_h_
#define _fdk_opts_h_

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

typedef struct MGHCBCT_Options_struct MGHCBCT_Options;
struct MGHCBCT_Options_struct {
    int first_img;
    int skip_img;
    int last_img;
    int resolution[3];
    float vol_size[3];
    float scale;
    char* input_dir;
    char* output_file;
};

#if defined __cplusplus
extern "C" {
#endif

void parse_args (MGHCBCT_Options* options, int argc, char* argv[]);

#if defined __cplusplus
}
#endif

#endif
