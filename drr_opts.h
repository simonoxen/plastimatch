/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _drr_opts_h_
#define _drr_opts_h_

#include "threading.h"

#define OPTION_RESOLUTION_STRING "resolution"
#define OPTION_RESOLUTION 'r'
#define OPTION_INPUT_FILE_STRING "input-file"
#define OPTION_INPUT_FILE 'I'
#define OPTION_OUTPUT_PREFIX_STRING "output-prefix"
#define OPTION_OUTPUT_PREFIX 'O'
#define OPTION_NUM_ANGLES_STRING "num-angles"
#define OPTION_NUM_ANGLES 'a'
#define OPTION_ANGLE_DIFF_STRING "angle-diff"
#define OPTION_ANGLE_DIFF 'A'
#define OPTION_SCALE_STRING "scale"
#define OPTION_SCALE 's'
//#define OPTION_TRUE_PGM_STRING "true-pgm"
//#define OPTION_TRUE_PGM 't'
#define OPTION_OUTPUT_FORMAT_STRING "output-format"
#define OPTION_OUTPUT_FORMAT 't'
#define OPTION_IMAGE_CENTER_STRING "image-center"
#define OPTION_IMAGE_CENTER 'c'
#define OPTION_IMAGE_SIZE_STRING "image-size"
#define OPTION_IMAGE_SIZE 'z'
#define OPTION_IMAGE_WINDOW_STRING "image-window"
#define OPTION_IMAGE_WINDOW 'w'
#define OPTION_INTERPOLATION_STRING "interpolation"
#define OPTION_INTERPOLATION 'i'
#define OPTION_ISOCENTER_STRING "isocenter"
#define OPTION_ISOCENTER 'o'
#define OPTION_MULTISPECTRAL_STRING "multispectral"
#define OPTION_MULTISPECTRAL 'S'
#define OPTION_EXPONENTIAL_STRING "exponential"
#define OPTION_EXPONENTIAL 'e'

enum drr_algorithm {
    DRR_ALGORITHM_EXACT,
    DRR_ALGORITHM_TRILINEAR_EXACT,
    DRR_ALGORITHM_TRILINEAR_APPROX,
    DRR_ALGORITHM_UNIFORM
};
typedef enum drr_algorithm Drr_algorithm;

#if defined (commentout)
#define INTERPOLATION_NONE                0
#define INTERPOLATION_TRILINEAR_EXACT     1
#define INTERPOLATION_TRILINEAR_APPROX    2
#endif

#define OUTPUT_FORMAT_PFM                 0
#define OUTPUT_FORMAT_PGM                 1
#define OUTPUT_FORMAT_RAW                 2

typedef struct drr_options Drr_options;
struct drr_options {
    Threading threading;
    int image_resolution[2];         /* In pixels */
    float image_size[2];             /* In mm */
    int have_image_center;           /* Was image_center spec'd in options? */
    float image_center[2];           /* In pixels */
    int have_image_window;           /* Was image_window spec'd in options? */
    int image_window[4];             /* In pixels */
    float isocenter[3];              /* In mm */

    int num_angles;
    int have_angle_diff;             /* Was angle_diff spec'd in options? */
    float angle_diff;                /* In degrees */

    int have_nrm;                    /* Was nrm specified? */
    float nrm[3];                    /* Normal vector (unitless) */
    float vup[3];                    /* Direction vector (unitless) */

    float sad;			     /* In mm */
    float sid;			     /* In mm */
    float scale;
    int exponential_mapping;
    int output_format;
    int multispectral;
    Drr_algorithm algorithm;
    char* input_file;
    char* output_prefix;
};

void parse_args (Drr_options* options, int argc, char* argv[]);

#endif
