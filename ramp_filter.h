/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ramp_filter_h_
#define _ramp_filter_h_

#if defined __cplusplus
extern "C" {
#endif

void RampFilter (unsigned short * data,
		 float* out,
		 unsigned int width,
		 unsigned int height);

#if defined __cplusplus
}
#endif

#endif
