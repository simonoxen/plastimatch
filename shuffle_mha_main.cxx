/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* Correct mha files which have incorrect patient orientations */
#include <stdio.h>
#include <stdlib.h>
#include "plm_config.h"
#include "volume.h"
#include "readmha.h"

int main (int argc, char* argv[])
{
    Volume* v_in;

    if (argc != 3 && argc != 4) {
	printf ("Usage: %s shuffle infile [outfile]\n", argv[0]);
	printf ("Shuffle value hard coded to flipping Z axis\n");
	exit (1);
    }

    return 0;
}
