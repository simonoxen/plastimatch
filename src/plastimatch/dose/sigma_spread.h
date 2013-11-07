#include <math.h>
#include <vector>
#include "volume.h"
#include "rpl_volume.h"
#include "proj_volume.h"

extern const double lookup_range_water[][2];
extern const double lookup_stop_water[][2];

extern const double lookup_r2_over_sigma2[][2];

void radiologic_length_to_sigma(std::vector<float>* p_sigma, std::vector<float>* p_density, float energy, float spacing_z, float sigma_src, float* sigma_max);

static float LR_interpolation(float density);
static float WER_interpolation(float density);

static double getrange(double energy);
static double getstop(double energy);


