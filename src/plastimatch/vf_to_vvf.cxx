/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
   Convert a mha vector field to varian format (for iLab workshop)
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "plmbase.h"

#include "volume.h"

void
print_usage (void)
{
    printf ("Usage: vf_to_vvf input_vf output_vvf\n");
    exit (1);
}

void
shuffle_vector_field (Volume* vol)
{
    int k1, k2, idx1, idx2;
    float *img_in, *img_out;

    img_in = (float*) vol->img;

    /* Allocate new array for output */
    img_out = (float*) malloc (vol->pix_size * vol->npix);

    /* Change RAI to RAS */
    for (k1 = 0; k1 < vol->dim[2]; k1++) {
	k2 = vol->dim[2] - k1 - 1;
	idx1 = k1 * vol->dim[0] * vol->dim[1];
	idx2 = k2 * vol->dim[0] * vol->dim[1];
	memcpy ((void*) &img_out[idx2], (void*) &img_in[idx1], vol->dim[0] * vol->dim[1] * vol->pix_size);
    }

    /* Swap img_ptrs */
    free (vol->img);
    vol->img = (void*) img_out;
}

#define WRITE_BLOCK (1024)
void
fwrite_block (void* buf, size_t size, size_t count, FILE* fp)
{
    size_t left_to_write = count * size;
    size_t cur = 0;
    char* bufc = (char*) buf;

    while (left_to_write > 0) {
	size_t this_write, rc;

	this_write = left_to_write;
	if (this_write > WRITE_BLOCK) this_write = WRITE_BLOCK;
	rc = fwrite (&bufc[cur], 1, this_write, fp);
	if (rc != this_write) {
	    fprintf (stderr, 
		     "Error writing to file.  rc=%lu, this_write=%lu\n",
		     (long unsigned) rc, (long unsigned) this_write);
	    return;
	}
	cur += rc;
	left_to_write -= rc;
    }
}

void
write_vvf (Volume* vol, char* vvf_out)
{
    FILE* fp;
    printf ("Writing vvf file...\n");
    fp = fopen (vvf_out, "wb");
    if (!fp) {
	fprintf (stderr, "Sorry, cannot open file %s for write\n", vvf_out);
	exit (-1);
    }
    /* Write num_pix */
    fwrite (vol->dim, sizeof(int), 3, fp);
    /* Write pixel spacing */
    fwrite (vol->pix_spacing, sizeof(float), 3, fp);
    /* Write vector field */
    fwrite (vol->img, sizeof(float), vol->npix * 3, fp);
    fclose (fp);
}

int
main (int argc, char *argv[])
{
    char *vf_in, *vvf_out;
    Volume* vol;

    if (argc != 3) {
	print_usage ();
    }
    vf_in = argv[1];
    vvf_out = argv[2];

    vol = read_mha (vf_in);
    if (!vol) {
	fprintf (stderr, "Sorry, couldn't open file \"%s\" for read.\n", vf_in);
	exit (-1);
    }

    if (vol->pix_type != PT_VF_FLOAT_INTERLEAVED) {
	fprintf (stderr, "Sorry, file \"%s\" is not an interleaved float vector field.\n", vf_in);
	fprintf (stderr, "Type = %d\n", vol->pix_type);
	exit (-1);
    }

#if defined (commentout)
    /* As of Oct 17 memo, we no longer have to shuffle */
    shuffle_vector_field (vol);
#endif

    write_vvf (vol, vvf_out);

    volume_destroy (vol);

    return 0;
}
