/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _demons_misc_h_
#define _demons_misc_h_

#if defined __cplusplus
extern "C" {
#endif
float* create_ker (float coeff, int half_width);
void validate_filter_widths (int *fw_out, int *fw_in);
void kernel_stats (float* kerx, float* kery, float* kerz, int fw[]);
#if defined __cplusplus
}
#endif

#endif
