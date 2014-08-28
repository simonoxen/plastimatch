/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _radiation_lut_h_
#define _radiation_lut_h_

#include <math.h>

float LR_interpolation(float density);
float WER_interpolation(float density);

double getrange(double energy);
double getstop(double energy);

extern const double lookup_proton_range_water[][2];
extern const double lookup_proton_stop_water[][2];

extern const double lookup_off_axis[][2];

#endif