/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rt_lut_h_
#define _rt_lut_h_

#include <math.h>

double getrange(double energy);
double getstop(double energy);

double compute_X0_from_HU(double CT_HU); // Radiation length

extern const double lookup_proton_range_water[][2];
extern const double lookup_proton_stop_water[][2];

extern const double lookup_off_axis[][2];

/* declaration of a matrix that contains the depth of the max for each energy */
extern const int max_depth_proton[];

/* declaration of a matrix that contains the alpha and p parameters of the particles (Range = f(E, alpha, p) */
extern const double particle_parameters[][2];

#endif
