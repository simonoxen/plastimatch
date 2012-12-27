/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "nkidecompress.h"

#include "file_util.h"
#include "nki_io.h"
#include "plm_int.h"
#include "print_and_exit.h"
#include "volume.h"

#define LINELEN 512

Volume* 
nki_load (const char* filename)
{
    char linebuf[LINELEN];

    FILE *fp = fopen (filename,"rb");
    if (!fp) {
	fprintf (stderr, "File %s not found\n", filename);
	return 0;
    }
    fprintf(stdout, "reading %s\n", filename);

    /* Find file size */   
    uint64_t fsize = file_size (filename);

    /* Make a buffer large enough to hold file */
    signed char *src = (signed char*) malloc (
        fsize * sizeof(signed char));

    /* Loop, reading lines from file */
    int dim1 = -1, dim2 = -1, dim3 = -1;
    bool have_start_pos = false;
    int junk;
    fpos_t pos;
    while (fgetpos (fp, &pos), fgets (linebuf, LINELEN, fp))
    {
	if (sscanf (linebuf, "dim1=%d", &dim1) == 1) {
            continue;
        }
	if (sscanf (linebuf, "dim2=%d", &dim2) == 1) {
            continue;
        }
	if (sscanf (linebuf, "dim3=%d", &dim3) == 1) {
            continue;
        }
	if (sscanf (linebuf, "nki_compression=%d", &junk) == 1) {
            /* End of ascii header, so back up, and find beginning 
               of compressed data. */
            fsetpos (fp, &pos);
            int prev = fgetc (fp);
            do {
                int curr = fgetc (fp);
                if (prev == 0x0c && curr == 0x0c) {
                    have_start_pos = true;
                    break;
                }
                prev = curr;
            } while (prev != EOF);

            /* Slurp up rest of file into a buffer */
            signed char *p = src;
            while (1) {
                size_t bytes_read = fread (p, 1, 2048, fp);
                if (bytes_read != 2048) break;
                p += bytes_read;
            }

            /* Done! */
            break;
        }
    }
    fclose (fp);

    if (dim1 == -1 || dim2 == -1 || dim3 == -1 || !have_start_pos) {
        print_and_exit ("Failure to parse NKI header\n");
    }

    short *dest = (short*) malloc (sizeof(short) * dim1 * dim2 * dim3);
    
    int rc = nki_private_decompress (dest, src, dim1 * dim2 * dim3);
    free (src);

    printf ("Decoded NKI size: %d %d %d, rc = %d\n", dim1, dim2, dim3, rc);

    if (rc == 0) {
        /* Decompression failure */
        free (dest);
        return 0;
    }

    Volume *vol  = new Volume;
    vol->pix_size = 2;
    vol->pix_type = PT_SHORT;
    vol->dim[0] = dim1;
    vol->dim[1] = dim2;
    vol->dim[2] = dim3;
    vol->offset[0] = 0;
    vol->offset[1] = 0;
    vol->offset[2] = 0;
    vol->spacing[0] = 1;
    vol->spacing[1] = 1;
    vol->spacing[2] = 1;
    vol->set_direction_cosines (0);
    vol->img = dest;

    return vol;
}
