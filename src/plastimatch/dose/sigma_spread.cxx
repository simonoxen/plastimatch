#include "sigma_spread.h"

void radiologic_length_to_sigma(std::vector<float>* p_sigma, std::vector<float>* p_density, float energy, float spacing_z, float sigma_src, float* sigma_max)
{
    std::vector<float> tmp_rglength (p_sigma->size(),0);

    int first_non_null_loc = 0;
    spacing_z = spacing_z/10; // converted to cm (the Highland formula is in cm!)

    /* initializiation of all the french_fries, except sigma, which is the output and calculate later */
    for(int i = 0; i < (int) p_sigma->size();i++)
    {
        if(i == 0)
        {
            tmp_rglength[i] = (*p_sigma)[i]; // remember that at this point french_fries_sigma is only a rglngth function without compensator
            (*p_sigma)[i] = 0;
        } 
        else 
        {
            tmp_rglength[i] = (*p_sigma)[i]-(*p_sigma)[i-1]; // rglength in the pixel
            (*p_sigma)[i] = 0;
        }
    }

    //We can now compute the sigma french_fries!!!!

    /* Step 1: the sigma is filled with zeros, so we let them at 0 as long as rg_length is 0, meaning the ray is out of the volume */
    /* we mark the first pixel in the volume, and the calculations will start with this one */
    for (int i = 0; i < (int) p_sigma->size(); i++)
    {
        if (tmp_rglength[i] > 0)
        {
            first_non_null_loc = i;
            break;
        }
        if (i == p_sigma->size())
        {
            first_non_null_loc = p_sigma->size()-1;
            printf("\n the french_fries is completely zeroed, the ray seems to not intersect the volume\n");
            return;
        }
    }

    /* Step 2: Each pixel in the volume will receive its sigma (in reality y0) value, according to the differential Highland formula */

    float energy_callback = energy;
    float mc2 = 939.4f;          /* proton mass at rest (MeV) */
    float c = 299792458.0f;        /* speed of light (m/s2) */

    float p = 0.0;              /* Proton momentum (passed in)          */
    float v = 0.0;              /* Proton velocity (passed in)          */
    float stop = 0;		/* stopping power energy (MeV.cm2/g) */
	
    float sum = 0.0;		/* integration expressions, right part of equation */
    float function_to_be_integrated; /* right term to be integrated in the Highland equation */
    float inverse_rad_length_integrated = 0; /* and left part */
    float y0 = 0;               /* y0 value used to update the french_fries values */
    float inv_rad_length;       /* 1/rad_length - used in the integration */
    float step;                 /* step of integration, will depends on the radiologic length */

    float POI_depth;            /* depth of the point of interest (where is calculated the sigma value)in cm - centered at the pixel center*/
    float pixel_depth;          /* depth of the contributing pixel to total sigma (in cm) - center between 2 pixels, the difference in rglength comes from the center of the previous pixel to the center of this pixel*/

    for (int i = first_non_null_loc; i < (int) p_sigma->size(); i++)
    {
        energy = energy_callback; // we reset the parameters for each sigma calculation
        sum = 0;
        inverse_rad_length_integrated = 0;

        POI_depth = (float) (i+0.5)*spacing_z;

        /*integration */
        for (int j = first_non_null_loc; j <= i && energy >0;j++)
        {

            /* p & v are updated */

            p= sqrt(2*energy*mc2+energy*energy)/c; // in MeV.s.m-1
            v= c*sqrt(1-pow((mc2/(energy+mc2)),2)); //in m.s-1

            if (i == j)
            {
                pixel_depth = (j+.25f)*spacing_z; // we integrate only up to the voxel center, not the whole pixel
                step = spacing_z/2;
            }
            else
            {
                pixel_depth = (j+0.5f)*spacing_z;
                step = spacing_z;
            }
            
            inv_rad_length = 1.0f / LR_interpolation((*p_density)[j]);

            function_to_be_integrated = (pow(((POI_depth - pixel_depth)/(p*v)),2) * inv_rad_length); //i in cm
            sum += function_to_be_integrated*step;

            inverse_rad_length_integrated += step * inv_rad_length;

            /* energy is updated after passing through dz */
            stop = (float) getstop(energy)* WER_interpolation((*p_density)[j]) * (*p_density)[j]; // dE/dx_mat = dE /dx_watter * WER * density (lut in g/cm2)
            energy = energy - stop*step;
        }

        if (energy  <= 0) // sigma formula is not defined anymore
        {
            return; // we can exit as the rest of the french_fries_sigma equals already 0
        }
        else // that means we reach the POI pixel and we can store the y0 value
        {
            (*p_sigma)[i] = 141.0f *(1.0f+1.0f/9.0f*log10(inverse_rad_length_integrated))* (float) sqrt(sum); // in mm
            if (*sigma_max < (*p_sigma)[i])
            {
                *sigma_max = (*p_sigma) [i];
            }
        }
    }
    return;
}

static float LR_interpolation(float density)
{
    return 36.08f*pow(density,-1.548765f); // in cm
}

static float WER_interpolation(float density) // interpolation between adip, water, muscle, PMMA and bone
{
    if (density <=1)
    {
        return 0.3825f * density + .6175f;
    }
    else if (density > 1 && density <=1.04)
    {
        return .275f * density + .725f;
    }
    else if (density > 1.04 && density <= 1.19)
    {
        return .1047f * density + .9021f;
    }
    else
    {
        return .0803f * density + .9311f;
    }
}

static double getrange(double energy)
{
    double energy1 = 0;
    double energy2 = 0;
    double range1 = 0;
    double range2 = 0;
    int i=0;

    if (energy >0)
    {
	while (energy >= energy1)
	{
            energy1 = lookup_range_water[i][0];
	    range1 = lookup_range_water[i][1];

	    if (energy >= energy1)
	    {
	    	energy2 = energy1;
		range2 = range1;
	    }
	    i++;
	}
	return (range2+(energy-energy2)*(range1-range2)/(energy1-energy2));
    }
    else
    {
	return 0;
    }
}

static double getstop(double energy)
{
    double energy1 = 0;
    double energy2 = 0;
    double stop1 = 0;
    double stop2 = 0;
    int i=0;

    if (energy >0)
    {
    	while (energy >= energy1)
	{
	    energy1 = lookup_stop_water[i][0];
	    stop1 = lookup_stop_water[i][1];
        
            if (energy >= energy1)
            {
		energy2 = energy1;
		stop2 = stop1;
	    }
	    i++;
	}
	return (stop2+(energy-energy2)*(stop1-stop2)/(energy1-energy2));
    }
    else
    {
	return 0;
    }
}

extern const double lookup_range_water[][2] ={
1.000E-03,	6.319E-06,
1.500E-03,	8.969E-06,	
2.000E-03,	1.137E-05,	
2.500E-03,	1.357E-05,	
3.000E-03,	1.560E-05,	
4.000E-03,	1.930E-05,	
5.000E-03,	2.262E-05,	
6.000E-03,	2.567E-05,	
7.000E-03,	2.849E-05,	
8.000E-03,	3.113E-05,	
9.000E-03,	3.363E-05,	
1.000E-02,	3.599E-05,	
1.250E-02,	4.150E-05,	
1.500E-02,	4.657E-05,	
1.750E-02,	5.131E-05,	
2.000E-02,	5.578E-05,	
2.250E-02,	6.005E-05,	
2.500E-02,	6.413E-05,	
2.750E-02,	6.806E-05,	
3.000E-02,	7.187E-05,	
3.500E-02,	7.916E-05,	
4.000E-02,	8.613E-05,	
4.500E-02,	9.284E-05,	
5.000E-02,	9.935E-05,	
5.500E-02,	1.057E-04,	
6.000E-02,	1.120E-04,	
6.500E-02,	1.182E-04,	
7.000E-02,	1.243E-04,	
7.500E-02,	1.303E-04,	
8.000E-02,	1.364E-04,	
8.500E-02,	1.425E-04,	
9.000E-02,	1.485E-04,	
9.500E-02,	1.546E-04,	
1.000E-01,	1.607E-04,	
1.250E-01,	1.920E-04,	
1.500E-01,	2.249E-04,	
1.750E-01,	2.598E-04,	
2.000E-01,	2.966E-04,	
2.250E-01,	3.354E-04,	
2.500E-01,	3.761E-04,	
2.750E-01,	4.186E-04,	
3.000E-01,	4.631E-04,	
3.500E-01,	5.577E-04,	
4.000E-01,	6.599E-04,	
4.500E-01,	7.697E-04,	
5.000E-01,	8.869E-04,	
5.500E-01,	1.012E-03,	
6.000E-01,	1.144E-03,	
6.500E-01,	1.283E-03,	
7.000E-01,	1.430E-03,	
7.500E-01,	1.584E-03,	
8.000E-01,	1.745E-03,	
8.500E-01,	1.913E-03,	
9.000E-01,	2.088E-03,	
9.500E-01,	2.270E-03,	
1.000E+00,	2.458E-03,	
1.250E+00,	3.499E-03,	
1.500E+00,	4.698E-03,	
1.750E+00,	6.052E-03,	
2.000E+00,	7.555E-03,	
2.250E+00,	9.203E-03,	
2.500E+00,	1.099E-02,	
2.750E+00,	1.292E-02,	
3.000E+00,	1.499E-02,	
3.500E+00,	1.952E-02,	
4.000E+00,	2.458E-02,	
4.500E+00,	3.015E-02,	
5.000E+00,	3.623E-02,	
5.500E+00,	4.279E-02,	
6.000E+00,	4.984E-02,	
6.500E+00,	5.737E-02,	
7.000E+00,	6.537E-02,	
7.500E+00,	7.384E-02,	
8.000E+00,	8.277E-02,	
8.500E+00,	9.215E-02,	
9.000E+00,	1.020E-01,	
9.500E+00,	1.123E-01,	
1.000E+01,	1.230E-01,	
1.250E+01,	1.832E-01,	
1.500E+01,	2.539E-01,	
1.750E+01,	3.350E-01,	
2.000E+01,	4.260E-01,	
2.500E+01,	6.370E-01,	
2.750E+01,	7.566E-01,	
3.000E+01,	8.853E-01,	
3.500E+01,	1.170E+00,	
4.000E+01,	1.489E+00,	
4.500E+01,	1.841E+00,	
5.000E+01,	2.227E+00,	
5.500E+01,	2.644E+00,	
6.000E+01,	3.093E+00,	
6.500E+01,	3.572E+00,	
7.000E+01,	4.080E+00,	
7.500E+01,	4.618E+00,	
8.000E+01,	5.184E+00,	
8.500E+01,	5.777E+00,	
9.000E+01,	6.398E+00,	
9.500E+01,	7.045E+00,	
1.000E+02,	7.718E+00,	
1.250E+02,	1.146E+01,	
1.500E+02,	1.577E+01,	
1.750E+02,	2.062E+01,	
2.000E+02,	2.596E+01,	
2.250E+02,	3.174E+01,	
2.500E+02,	3.794E+01,	
2.750E+02,	4.452E+01,	
3.000E+02,	5.145E+01,	
3.500E+02,	6.628E+01,	
4.000E+02,	8.225E+01,	
4.500E+02,	9.921E+01,	
5.000E+02,	1.170E+02,	
5.500E+02,	1.356E+02,	
6.000E+02,	1.549E+02,	
6.500E+02,	1.747E+02,	
7.000E+02,	1.951E+02,	
7.500E+02,	2.159E+02,	
8.000E+02,	2.372E+02,	
8.500E+02,	2.588E+02,	
9.000E+02,	2.807E+02,	
9.500E+02,	3.029E+02,	
1.000E+03,	3.254E+02,	
1.500E+03,	5.605E+02,	
2.000E+03,	8.054E+02,	
2.500E+03,	1.054E+03,	
3.000E+03,	1.304E+03,	
4.000E+03,	1.802E+03,	
5.000E+03,	2.297E+03,	
6.000E+03,	2.787E+03,	
7.000E+03,	3.272E+03,	
8.000E+03,	3.752E+03,	
9.000E+03,	4.228E+03,	
1.000E+04,	4.700E+03,	
};

extern const double lookup_stop_water[][2] ={
0.001,	176.9,
0.0015,	198.4,
0.002,	218.4,
0.0025,	237,
0.003,	254.4,
0.004,	286.4,
0.005,	315.3,
0.006,	342,
0.007,	366.7,
0.008,	390,
0.009,	412,
0.01,	432.9,
0.0125,	474.5,
0.015,	511,
0.0175,	543.7,
0.02,	573.3,
0.0225,	600.1,
0.025,	624.5,
0.0275,	646.7,
0.03,	667.1,
0.035,	702.8,
0.04,	732.4,
0.045,	756.9,
0.05,	776.8,
0.055,	792.7,
0.06,	805,
0.065,	814.2,
0.07,	820.5,
0.075,	824.3,
0.08,	826,
0.085,	825.8,
0.09,	823.9,
0.095,	820.6,
0.1,	816.1,
0.125,	781.4,
0.15,	737.1,
0.175,	696.9,
0.2,	661.3,
0.225,	629.4,
0.25,	600.6,
0.275,	574.4,
0.3,	550.4,
0.35,	508,
0.4,	471.9,
0.45,	440.6,
0.5,	413.2,
0.55,	389.1,
0.6,	368,
0.65,	349.2,
0.7,	332.5,
0.75,	317.5,
0.8,	303.9,
0.85,	291.7,
0.9,	280.5,
0.95,	270.2,
1,	260.8,
1.25,	222.9,
1.5,	195.7,
1.75,	174.9,
2,	158.6,
2.25,	145.4,
2.5,	134.4,
2.75,	125.1,
3,	117.2,
3.5,	104.2,
4,	94.04,
4.5,	85.86,
5,	79.11,
5.5,	73.43,
6,	68.58,
6.5,	64.38,
7,	60.71,
7.5,	57.47,
8,	54.6,
8.5,	52.02,
9,	49.69,
9.5,	47.59,
10,	45.67,
12.5,	38.15,
15,	32.92,
17.5,	29.05,
20,	26.07,
25,	21.75,
27.5,	20.13,
30,	18.76,
35,	16.56,
40,	14.88,
45,	13.54,
50,	12.45,
55,	11.54,
60,	10.78,
65,	10.13,
70,	9.559,
75,	9.063,
80,	8.625,
85,	8.236,
90,	7.888,
95,	7.573,
100,	7.289,
125,	6.192,
150,	5.445,
175,	4.903,
200,	4.492,
225,	4.17,
250,	3.911,
275,	3.698,
300,	3.52,
350,	3.241,
400,	3.032,
450,	2.871,
500,	2.743,
550,	2.64,
600,	2.556,
650,	2.485,
700,	2.426,
750,	2.376,
800,	2.333,
850,	2.296,
900,	2.264,
950,	2.236,
1000,	2.211,
1500,	2.07,
2000,	2.021,
2500,	2.004,
3000,	2.001,
4000,	2.012,
5000,	2.031,
6000,	2.052,
7000,	2.072,
8000,	2.091,
9000,	2.109,
10000,	2.126,
};

extern const double lookup_r2_over_sigma2[][2]={
0.00, 1.0000,
0.01, 1.0000,
0.02, 0.9998,
0.03, 0.9996,
0.04, 0.9992,
0.05, 0.9988,
0.06, 0.9982,
0.07, 0.9976,
0.08, 0.9968,
0.09, 0.9960,
0.10, 0.9950,
0.11, 0.9940,
0.12, 0.9928,
0.13, 0.9916,
0.14, 0.9902,
0.15, 0.9888,
0.16, 0.9873,
0.17, 0.9857,
0.18, 0.9839,
0.19, 0.9821,
0.20, 0.9802,
0.21, 0.9782,
0.22, 0.9761,
0.23, 0.9739,
0.24, 0.9716,
0.25, 0.9692,
0.26, 0.9668,
0.27, 0.9642,
0.28, 0.9616,
0.29, 0.9588,
0.30, 0.9560,
0.31, 0.9531,
0.32, 0.9501,
0.33, 0.9470,
0.34, 0.9438,
0.35, 0.9406,
0.36, 0.9373,
0.37, 0.9338,
0.38, 0.9303,
0.39, 0.9268,
0.40, 0.9231,
0.41, 0.9194,
0.42, 0.9156,
0.43, 0.9117,
0.44, 0.9077,
0.45, 0.9037,
0.46, 0.8996,
0.47, 0.8954,
0.48, 0.8912,
0.49, 0.8869,
0.50, 0.8825,
0.51, 0.8781,
0.52, 0.8735,
0.53, 0.8690,
0.54, 0.8643,
0.55, 0.8596,
0.56, 0.8549,
0.57, 0.8501,
0.58, 0.8452,
0.59, 0.8403,
0.60, 0.8353,
0.61, 0.8302,
0.62, 0.8251,
0.63, 0.8200,
0.64, 0.8148,
0.65, 0.8096,
0.66, 0.8043,
0.67, 0.7990,
0.68, 0.7936,
0.69, 0.7882,
0.70, 0.7827,
0.71, 0.7772,
0.72, 0.7717,
0.73, 0.7661,
0.74, 0.7605,
0.75, 0.7548,
0.76, 0.7492,
0.77, 0.7435,
0.78, 0.7377,
0.79, 0.7319,
0.80, 0.7261,
0.81, 0.7203,
0.82, 0.7145,
0.83, 0.7086,
0.84, 0.7027,
0.85, 0.6968,
0.86, 0.6909,
0.87, 0.6849,
0.88, 0.6790,
0.89, 0.6730,
0.90, 0.6670,
0.91, 0.6610,
0.92, 0.6549,
0.93, 0.6489,
0.94, 0.6429,
0.95, 0.6368,
0.96, 0.6308,
0.97, 0.6247,
0.98, 0.6187,
0.99, 0.6126,
1.00, 0.6065,
1.01, 0.6005,
1.02, 0.5944,
1.03, 0.5883,
1.04, 0.5823,
1.05, 0.5762,
1.06, 0.5702,
1.07, 0.5641,
1.08, 0.5581,
1.09, 0.5521,
1.10, 0.5461,
1.11, 0.5401,
1.12, 0.5341,
1.13, 0.5281,
1.14, 0.5222,
1.15, 0.5162,
1.16, 0.5103,
1.17, 0.5044,
1.18, 0.4985,
1.19, 0.4926,
1.20, 0.4868,
1.21, 0.4809,
1.22, 0.4751,
1.23, 0.4693,
1.24, 0.4636,
1.25, 0.4578,
1.26, 0.4521,
1.27, 0.4464,
1.28, 0.4408,
1.29, 0.4352,
1.30, 0.4296,
1.31, 0.4240,
1.32, 0.4184,
1.33, 0.4129,
1.34, 0.4075,
1.35, 0.4020,
1.36, 0.3966,
1.37, 0.3912,
1.38, 0.3859,
1.39, 0.3806,
1.40, 0.3753,
1.41, 0.3701,
1.42, 0.3649,
1.43, 0.3597,
1.44, 0.3546,
1.45, 0.3495,
1.46, 0.3445,
1.47, 0.3394,
1.48, 0.3345,
1.49, 0.3295,
1.50, 0.3247,
1.51, 0.3198,
1.52, 0.3150,
1.53, 0.3102,
1.54, 0.3055,
1.55, 0.3008,
1.56, 0.2962,
1.57, 0.2916,
1.58, 0.2870,
1.59, 0.2825,
1.60, 0.2780,
1.61, 0.2736,
1.62, 0.2692,
1.63, 0.2649,
1.64, 0.2606,
1.65, 0.2563,
1.66, 0.2521,
1.67, 0.2480,
1.68, 0.2439,
1.69, 0.2398,
1.70, 0.2357,
1.71, 0.2318,
1.72, 0.2278,
1.73, 0.2239,
1.74, 0.2201,
1.75, 0.2163,
1.76, 0.2125,
1.77, 0.2088,
1.78, 0.2051,
1.79, 0.2015,
1.80, 0.1979,
1.81, 0.1944,
1.82, 0.1909,
1.83, 0.1874,
1.84, 0.1840,
1.85, 0.1806,
1.86, 0.1773,
1.87, 0.1740,
1.88, 0.1708,
1.89, 0.1676,
1.90, 0.1645,
1.91, 0.1614,
1.92, 0.1583,
1.93, 0.1553,
1.94, 0.1523,
1.95, 0.1494,
1.96, 0.1465,
1.97, 0.1436,
1.98, 0.1408,
1.99, 0.1381,
2.00, 0.1353,
2.01, 0.1326,
2.02, 0.1300,
2.03, 0.1274,
2.04, 0.1248,
2.05, 0.1223,
2.06, 0.1198,
2.07, 0.1174,
2.08, 0.1150,
2.09, 0.1126,
2.10, 0.1103,
2.11, 0.1080,
2.12, 0.1057,
2.13, 0.1035,
2.14, 0.1013,
2.15, 0.0991,
2.16, 0.0970,
2.17, 0.0949,
2.18, 0.0929,
2.19, 0.0909,
2.20, 0.0889,
2.21, 0.0870,
2.22, 0.0851,
2.23, 0.0832,
2.24, 0.0814,
2.25, 0.0796,
2.26, 0.0778,
2.27, 0.0760,
2.28, 0.0743,
2.29, 0.0727,
2.30, 0.0710,
2.31, 0.0694,
2.32, 0.0678,
2.33, 0.0662,
2.34, 0.0647,
2.35, 0.0632,
2.36, 0.0617,
2.37, 0.0603,
2.38, 0.0589,
2.39, 0.0575,
2.40, 0.0561,
2.41, 0.0548,
2.42, 0.0535,
2.43, 0.0522,
2.44, 0.0510,
2.45, 0.0497,
2.46, 0.0485,
2.47, 0.0473,
2.48, 0.0462,
2.49, 0.0450,
2.50, 0.0439,
2.51, 0.0428,
2.52, 0.0418,
2.53, 0.0407,
2.54, 0.0397,
2.55, 0.0387,
2.56, 0.0377,
2.57, 0.0368,
2.58, 0.0359,
2.59, 0.0349,
2.60, 0.0340,
2.61, 0.0332,
2.62, 0.0323,
2.63, 0.0315,
2.64, 0.0307,
2.65, 0.0299,
2.66, 0.0291,
2.67, 0.0283,
2.68, 0.0276,
2.69, 0.0268,
2.70, 0.0261,
2.71, 0.0254,
2.72, 0.0247,
2.73, 0.0241,
2.74, 0.0234,
2.75, 0.0228,
2.76, 0.0222,
2.77, 0.0216,
2.78, 0.0210,
2.79, 0.0204,
2.80, 0.0198,
2.81, 0.0193,
2.82, 0.0188,
2.83, 0.0182,
2.84, 0.0177,
2.85, 0.0172,
2.86, 0.0167,
2.87, 0.0163,
2.88, 0.0158,
2.89, 0.0154,
2.90, 0.0149,
2.91, 0.0145,
2.92, 0.0141,
2.93, 0.0137,
2.94, 0.0133,
2.95, 0.0129,
2.96, 0.0125,
2.97, 0.0121,
2.98, 0.0118,
2.99, 0.0114,
3.00, 0.0111,
};