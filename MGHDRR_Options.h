#pragma once

typedef struct MGHDRR_Options_struct MGHDRR_Options;
struct MGHDRR_Options_struct {
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
    float sad;			     /* In mm */
    float sid;			     /* In mm */
    float scale;
    int exponential_mapping;
    //int true_pgm;
    char* output_format;
    int multispectral;
    int interpolation;
    char* input_file;
    char* output_prefix;
	char* ProjAngle_file;
};