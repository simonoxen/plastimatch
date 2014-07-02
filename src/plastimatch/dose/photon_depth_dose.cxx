/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "Photon_depth_dose.h"

Photon_depth_dose::Photon_depth_dose ()
{
    this->d_lut = NULL;
    this->e_lut = NULL;

    this->E0 = 0.0;
    this->spread = 0.0;
    this->dres = 1.0;
    this->dmax = 0.0;
    this->weight = 1.0;

    this->num_samples = 0;
}

Photon_depth_dose::Photon_depth_dose (
    double E0, double spread, double dres, double dmax, double weight)
{
    this->d_lut = NULL;
    this->e_lut = NULL;

    this->E0 = E0;
    this->spread = spread;
    this->dres = dres;
    this->dmax = dmax;
    this->weight = weight;

    this->generate();
}

Photon_depth_dose::~Photon_depth_dose ()
{
    if (this->d_lut) {
        free (this->d_lut);
    }
    if (this->e_lut) {
        free (this->e_lut);
    }
}

bool
Photon_depth_dose::load (const char* fn)
{
    FILE* fp = fopen (fn, "r");
    char linebuf[128];

    if (!fp) {
        return false;
    }

    fgets (linebuf, 128, fp);
    fclose (fp);

    if (!strncmp (linebuf, "00001037", strlen ("00001037"))) {
        return this->load_xio (fn);
    } else {
        return this->load_txt (fn);
    }
}

bool
Photon_depth_dose::load_xio (const char* fn)
{
    int i, j;
    char* ptoken;
    char linebuf[128];
    FILE* fp = fopen (fn, "r");

    // Need to check for a magic number (00001037) here?
    
    /* skip the first 4 lines */
    for (i=0; i<4; i++) {
        fgets (linebuf, 128, fp);
    }

    /* line 5 contains the # of samples */
    fgets (linebuf, 128, fp);
    sscanf (linebuf, "%i", &this->num_samples);

    this->d_lut = (float*)malloc (this->num_samples*sizeof(float));
    this->e_lut = (float*)malloc (this->num_samples*sizeof(float));
    
    memset (this->d_lut, 0, this->num_samples*sizeof(float));
    memset (this->e_lut, 0, this->num_samples*sizeof(float));

    /* load in the depths (10 samples per line) */
    for (i=0, j=0; i<(this->num_samples/10)+1; i++) {
        fgets (linebuf, 128, fp);
        ptoken = strtok (linebuf, ",\n\0");
        while (ptoken) {
            this->d_lut[j++] = (float) strtod (ptoken, NULL);
            ptoken = strtok (NULL, ",\n\0");
        }
    }
    this->dmax = this->d_lut[j-1];

    /* load in the energies (10 samples per line) */
    for (i=0, j=0; i<(this->num_samples/10)+1; i++) {
        fgets (linebuf, 128, fp);
        ptoken = strtok (linebuf, ",\n\0");
        while (ptoken) {
            this->e_lut[j] = (float) strtod (ptoken, NULL);
            ptoken = strtok (NULL, ",\n\0");
            j++;
        }
    }

    fclose (fp);
    return true;
}

bool
Photon_depth_dose::load_txt (const char* fn)
{
    char linebuf[128];
    FILE* fp = fopen (fn, "r");

    while (fgets (linebuf, 128, fp)) {
        float range, dose;

        if (2 != sscanf (linebuf, "%f %f", &range, &dose)) {
            break;
        }

        this->num_samples++;
        this->d_lut = (float*) realloc (
                        this->d_lut,
                        this->num_samples * sizeof(float));

        this->e_lut = (float*) realloc (
                        this->e_lut,
                        this->num_samples * sizeof(float));

        this->d_lut[this->num_samples-1] = range;
        this->e_lut[this->num_samples-1] = dose;
        this->dmax = range;         /* Assume entries are sorted */
    }

    fclose (fp);
    return true;
}

bool
Photon_depth_dose::generate ()
{
    int i;
    double d;

#if SPECFUN_FOUND
    if (!this->E0) {
        printf ("ERROR: Failed to generate beam -- energy not specified.\n");
        return false;
    }
    if (!this->spread) {
        printf ("ERROR: Failed to generate beam -- energy spread not specified.\n");
        return false;
    }
    if (!this->dmax) {
        printf ("ERROR: Failed to generate beam -- max depth not specified.\n");
        return false;
    }

    this->num_samples = (int) floorf (this->dmax / this->dres);

    this->d_lut = (float*) malloc (this->num_samples*sizeof(float));
    this->e_lut = (float*) malloc (this->num_samples*sizeof(float));
    
    memset (this->d_lut, 0, this->num_samples*sizeof(float));
    memset (this->e_lut, 0, this->num_samples*sizeof(float));

    for (d=0, i=0; i<this->num_samples; d+=this->dres, i++) {
        d_lut[i] = d;
        e_lut[i] = photon_curve(d);
    }

    return true;
#else
    printf ("ERROR: No specfun found.\n");
    return false;
#endif
}

void
Photon_depth_dose::dump (const char* fn) const
{
    FILE* fp = fopen (fn, "w");

    for (int i=0; i<this->num_samples; i++) {
       fprintf (fp, "%3.2f %3.2f\n", this->d_lut[i], this->e_lut[i]);
    }

    fclose (fp);
}

float
Photon_depth_dose::lookup_energy (float depth) const
{	
	int i;
    float energy = 0.0f;

    /* Sanity check */
    if (depth < 0) {
        return 0.0f;
    }

    /* Find index into profile arrays */
    for (i = 0; i < this->num_samples-1; i++) {
        if (this->d_lut[i] > depth) {
            i--;
            break;
        }
    }

    /* Clip input depth to maximum in lookup table */
    if (i == this->num_samples-1) {
        depth = this->d_lut[i];
    }

    /* Use index to lookup and interpolate energy */
    if (i >= 0 || i < this->num_samples) {
        // linear interpolation
        energy = this->e_lut[i]
                 + (depth - this->d_lut[i])
                 * ((this->e_lut[i+1] - this->e_lut[i]) 
                 / (this->d_lut[i+1] - this->d_lut[i]));
    } else {
        // we wen't past the end of the lookup table
        energy = 0.0f;
    }

    return energy;   
}

double photon_curve(const double d){
	int low = 0;
	double rest = 0;
	if (d <0 || d >400)
	{
		return 0;
	}
	else
	{
		low = (int) d;
		rest = d - (double) low;
		return photon_depth_dose_lut[low]+ rest * ( photon_depth_dose_lut[low+1] - photon_depth_dose_lut[low] );
	}
}

extern const double photon_depth_dose_lut[]={
17.6,
40.8,
60.0,
74.9,
86.7,
94.7,
98.3,
99.1,
100.0,
99.6,
99.4,
99.6,
99.5,
99.6,
99.2,
98.9,
99.0,
99.2,
99.2,
98.2,
97.0,
96.4,
97.2,
97.5,
97.2,
96.8,
96.2,
95.9,
96.5,
96.8,
95.6,
95.0,
94.7,
94.2,
94.0,
93.7,
94.5,
93.8,
93.2,
93.6,
92.4,
93.4,
92.8,
92.2,
92.4,
92.0,
90.7,
91.8,
91.7,
92.0,
91.3,
91.0,
90.3,
90.0,
91.2,
90.5,
89.0,
89.1,
89.6,
89.9,
89.2,
89.0,
88.7,
88.3,
88.6,
88.0,
87.7,
87.3,
87.3,
87.3,
86.0,
86.0,
86.2,
85.6,
85.8,
85.3,
85.2,
85.4,
85.6,
85.5,
85.1,
84.4,
84.3,
84.3,
83.4,
83.2,
82.4,
83.0,
82.9,
82.6,
82.7,
82.7,
82.5,
82.2,
82.1,
82.1,
81.1,
80.6,
80.4,
80.6,
80.1,
79.7,
79.0,
79.0,
78.3,
78.5,
78.8,
78.5,
78.1,
77.5,
78.6,
78.4,
78.1,
76.9,
77.3,
77.1,
76.5,
76.5,
75.7,
75.6,
74.5,
74.6,
74.3,
74.6,
74.4,
74.2,
73.9,
73.8,
73.7,
73.7,
72.8,
73.6,
72.3,
72.0,
72.4,
72.3,
72.1,
71.6,
71.2,
71.3,
71.1,
70.9,
70.1,
69.8,
69.6,
69.1,
68.8,
68.4,
68.3,
68.8,
68.3,
68.6,
68.0,
67.1,
67.2,
67.1,
66.6,
67.2,
66.7,
65.3,
66.0,
64.8,
65.1,
64.8,
64.0,
64.3,
64.0,
63.4,
63.9,
64.0,
63.6,
63.2,
63.0,
62.8,
63.0,
63.0,
62.2,
60.9,
61.2,
60.8,
60.4,
61.4,
61.2,
61.1,
60.7,
60.7,
60.1,
59.7,
59.4,
58.0,
58.5,
57.9,
58.0,
58.0,
57.6,
57.3,
57.0,
56.2,
56.9,
57.1,
56.2,
56.8,
55.7,
55.8,
55.7,
55.2,
56.1,
55.4,
55.0,
55.2,
54.6,
54.5,
54.4,
53.4,
53.6,
54.0,
53.5,
53.4,
52.9,
52.9,
52.8,
52.3,
51.8,
51.9,
52.3,
52.0,
52.0,
51.5,
51.1,
50.7,
50.7,
50.1,
49.9,
50.3,
49.8,
49.7,
49.4,
49.6,
49.1,
49.0,
49.4,
48.9,
49.2,
48.9,
48.4,
48.0,
47.6,
47.2,
46.8,
46.9,
47.5,
47.0,
47.1,
46.6,
46.3,
46.0,
46.8,
45.9,
45.7,
45.0,
44.9,
44.9,
44.5,
44.3,
43.6,
43.9,
43.5,
43.4,
43.6,
43.4,
43.1,
43.5,
42.9,
43.2,
42.6,
42.1,
41.9,
41.6,
41.7,
41.7,
42.0,
41.7,
41.7,
40.7,
41.0,
40.7,
39.9,
40.5,
40.2,
40.6,
39.8,
39.3,
39.2,
38.9,
39.0,
38.9,
38.5,
38.6,
38.5,
38.8,
38.5,
37.6,
37.6,
37.3,
37.1,
36.7,
36.3,
36.7,
36.8,
36.8,
36.1,
36.3,
36.7,
36.0,
35.0,
35.0,
35.8,
35.3,
35.2,
34.8,
35.3,
34.6,
34.3,
33.7,
33.8,
33.8,
33.9,
33.9,
33.8,
34.0,
33.8,
32.9,
32.8,
33.0,
32.3,
32.1,
32.1,
32.3,
32.7,
32.4,
32.5,
32.1,
32.2,
32.0,
32.0,
31.7,
31.5,
31.4,
30.5,
30.7,
30.9,
30.5,
30.6,
30.6,
29.6,
30.1,
29.8,
30.3,
29.8,
29.5,
30.4,
29.2,
29.2,
29.1,
29.4,
28.6,
28.5,
28.6,
28.4,
28.5,
28.3,
28.3,
28.2,
27.7,
27.4,
27.1,
27.1,
26.9,
27.1,
26.7,
26.6,
26.8,
26.5,
26.2,
26.3,
25.9,
26.1,
25.7,
25.5,
25.7,
25.0,
24.9,
24.7,
24.7,
25.0,
24.5,
24.3,
24.2,
23.9,
22.3,
22.3,
};

