/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined (WIN32)
#include <direct.h>
#define mkdir(a,b) _mkdir(a)
#define PATH_MAX _MAX_PATH
#else
#include <limits.h>
#endif

#include "getopt.h"
#include "mha_io.h"

class Program_Parms {
public:
    char mha_in_fn[PATH_MAX];
    char rtog_dose_out_fn[PATH_MAX];
    float scale;
public:
    Program_Parms () {
	*mha_in_fn = 0;
	*rtog_dose_out_fn = 0;
	scale = 1.0;
    }
};

void
print_usage (void)
{
    printf ("Usage: mha_to_rtog_dose [options]\n");
    printf ("  -i filename    input file name\n");
    printf ("  -o filename    output file name\n");
    printf ("  -s float       rtog dose scale\n");
    exit (0);
}

void
parse_args (Program_Parms* parms, int argc, char* argv[])
{
    int ch, rc;
    static struct option longopts[] = {
	{ "input",             required_argument,      NULL,           'i' },
	{ "output",            required_argument,      NULL,           'o' },
	{ "scale",             required_argument,      NULL,           's' },
	{ NULL,                    0,                      NULL,           0 }
    };

    while ((ch = getopt_long (argc, argv, "i:n:o:s:", longopts, NULL))) {
	if (ch == -1) break;
	switch (ch) {
	case 'i':
	    strncpy (parms->mha_in_fn, optarg, PATH_MAX);
	    break;
	case 'o':
	    strncpy (parms->rtog_dose_out_fn, optarg, PATH_MAX);
	    break;
	case 's':
	    rc = sscanf (optarg, "%g", &(parms->scale));
	    if (rc != 1) {
		printf ("RTOG dose scale value option must have one arguments\n");
		exit (1);
	    }
	    break;
	default:
	    print_usage ();
	    break;
	}
    }
}

void
convert_dose (Program_Parms* parms)
{
    Volume* vol;
    FILE* fp;
    int i, slice_voxels;
    unsigned short *img_us, *p;
    float *img_f;

    vol = read_mha (parms->mha_in_fn);
    if (!vol) exit (-1);

    printf ("Scaling and converting (%d pix)...\n", vol->npix);
    img_f = (float*) vol->img;
    img_us = (unsigned short*) malloc (vol->npix * sizeof(unsigned short));
    for (i = 0; i < vol->npix; i++) {
	float raw = img_f[i];

	/* correct for scale */
	unsigned short raw_us = (unsigned short) (raw / parms->scale);

	/* swap bytes */
	unsigned short byte1 = (raw_us & 0xFF00) >> 8;
	unsigned short byte2 = (raw_us & 0x00FF) << 8;
	raw_us = byte1 | byte2;
	img_us[i] = raw_us;
    }

    printf ("Writing...\n");
    fp = fopen (parms->rtog_dose_out_fn, "wb");
    slice_voxels = vol->dim[0] * vol->dim[1];
    p = img_us + vol->npix;
    for (i = 0; i < vol->dim[2]; i++) {
	p -= slice_voxels;
	fwrite (p, sizeof(unsigned short), slice_voxels, fp);
    }
    fclose (fp);

    free (img_us);
    free (vol->img);
    free (vol);
}

int
main (int argc, char* argv[])
{
    Program_Parms parms;

    parse_args (&parms, argc, argv);

    convert_dose (&parms);

    return 0;
}
