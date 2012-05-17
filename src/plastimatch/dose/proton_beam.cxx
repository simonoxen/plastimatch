/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "plmdose.h"

Proton_Beam::Proton_Beam ()
{
    memset (this->src, 0, 3*sizeof (double));
    memset (this->isocenter, 0, 3*sizeof (double));

    this->d_lut = NULL;
    this->e_lut = NULL;

    this->E0 = 0.0;
    this->spread = 0.0;
    this->dmax = 0.0;
    this->dres = 0.0;
    this->num_samples = 0;
}

Proton_Beam::~Proton_Beam ()
{
    if (this->d_lut) {
        free (this->d_lut);
    }
    if (this->e_lut) {
        free (this->e_lut);
    }
}

void
Proton_Beam::load (const char* fn)
{
    FILE* fp = fopen (fn, "r");
    char linebuf[128];

    if (!fp) {
        printf ("Error reading bragg curve data from file.\n");
        printf ("Terminating...\n");
        exit(0);
    }

    fgets (linebuf, 128, fp);
    fclose (fp);

    if (!strncmp (linebuf, "00001037", strlen ("00001037"))) {
        this->load_xio (fn);
    } else {
//        this->load_txt (fn);
    }
}

void
Proton_Beam::load_xio (const char* fn)
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
}
