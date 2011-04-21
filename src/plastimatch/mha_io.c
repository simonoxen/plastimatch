/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "plm_path.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if (defined(_WIN32) || defined(WIN32))
#include <io.h>        // windows //
#endif
#include "fwrite_block.h"
#include "mha_io.h"
#include "string_util.h"
#include "volume.h"

#define LINELEN 128
#define MIN_SHORT -32768
#define MAX_SHORT 32767

/* -----------------------------------------------------------------------
   Private functions
   ----------------------------------------------------------------------- */
static void 
write_mha_internal (
    const char* filename,    /* Input: filename to write to */
    Volume* vol,             /* Input: volume to write */
    int mh5                  /* Input: force 512 byte header */
)
{
    FILE* fp;
    char* mha_header = 
	"ObjectType = Image\n"
	"NDims = 3\n"
	"BinaryData = True\n"
	"BinaryDataByteOrderMSB = False\n"
	"TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
	"Offset = %g %g %g\n"
	"CenterOfRotation = 0 0 0\n"
	"ElementSpacing = %g %g %g\n"
	"DimSize = %d %d %d\n"
	"AnatomicalOrientation = RAI\n"
	"%s"
	"ElementType = %s\n"
	"ElementDataFile = LOCAL\n";
    char* element_type;
    int bytes_written;

    if (vol->pix_type == PT_VF_FLOAT_PLANAR) {
	fprintf (stderr, "Error, PT_VF_FLOAT_PLANAR not implemented\n");
	exit (-1);
    }
    fp = fopen (filename,"wb");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for write\n", filename);
	return;
    }
    switch (vol->pix_type) {
    case PT_UCHAR:
	element_type = "MET_UCHAR";
	break;
    case PT_SHORT:
	element_type = "MET_SHORT";
	break;
    case PT_UINT32:
	element_type = "MET_UINT";
	break;
    case PT_FLOAT:
	element_type = "MET_FLOAT";
	break;
    case PT_VF_FLOAT_INTERLEAVED:
	element_type = "MET_FLOAT";
	break;
    case PT_VF_FLOAT_PLANAR:
    default:
	fprintf (stderr, "Unhandled type in write_mha().\n");
	exit (-1);
    }
    bytes_written = fprintf (fp, mha_header, 
	     vol->offset[0], vol->offset[1], vol->offset[2], 
	     vol->pix_spacing[0], vol->pix_spacing[1], vol->pix_spacing[2], 
	     vol->dim[0], vol->dim[1], vol->dim[2],
	     (vol->pix_type == PT_VF_FLOAT_INTERLEAVED) 
	     ? "ElementNumberOfChannels = 3\n" : "",
	     element_type);
    fflush (fp);

    if (mh5) {
	while (bytes_written < 512) {
	    fprintf(fp,"\n");
	    bytes_written ++;
	}
    }

    fwrite_block (vol->img, vol->pix_size, vol->npix, fp);

    fclose (fp);
}

static Volume* 
read_mha_internal (
    const char* filename,    /* Input: filename to read from */
    int mh5                  /* Input: force 512 byte header */
)
{
    int rc;
    char linebuf[LINELEN];
    Volume* vol;
    int tmp;
    FILE* fp;

    fp = fopen (filename,"rb");
    if (!fp) {
	fprintf (stderr, "File %s not found\n", filename);
	return 0;
    }
   
    fprintf(stdout, "reading %s\n", filename);
 
    vol  = (Volume*) malloc (sizeof(Volume));
    vol->pix_size = -1;
    vol->pix_type = PT_UNDEFINED;
    while (fgets (linebuf, LINELEN, fp)) {
	string_util_rtrim_whitespace (linebuf);
	if (strcmp (linebuf, "ElementDataFile = LOCAL") == 0) {
	    break;
	}
	if (sscanf (linebuf, "DimSize = %d %d %d",
		&vol->dim[0],
		&vol->dim[1],
		&vol->dim[2]) == 3) {
	    vol->npix = vol->dim[0] * vol->dim[1] * vol->dim[2];
	    continue;
	}
	if (sscanf (linebuf, "Offset = %g %g %g",
		&vol->offset[0],
		&vol->offset[1],
		&vol->offset[2]) == 3) {
	    continue;
	}
	if (sscanf (linebuf, "ElementSpacing = %g %g %g",
		&vol->pix_spacing[0],
		&vol->pix_spacing[1],
		&vol->pix_spacing[2]) == 3) {
	    continue;
	}
	if (sscanf (linebuf, "TransformMatrix = %g %g %g %g %g %g %g %g %g",
		&vol->direction_cosines[0],
		&vol->direction_cosines[1],
		&vol->direction_cosines[2],
		&vol->direction_cosines[3],
		&vol->direction_cosines[4],
		&vol->direction_cosines[5],
		&vol->direction_cosines[6],
		&vol->direction_cosines[7],
		&vol->direction_cosines[8]) == 9) {
	    continue;
	}
	if (sscanf (linebuf, "ElementNumberOfChannels = %d", &tmp) == 1) {
	    if (vol->pix_type == PT_UNDEFINED || vol->pix_type == PT_FLOAT) {
		vol->pix_type = PT_VF_FLOAT_INTERLEAVED;
		vol->pix_size = 3*sizeof(float);
	    }
	    continue;
	}
	if (strcmp (linebuf, "ElementType = MET_FLOAT") == 0) {
	    if (vol->pix_type == PT_UNDEFINED) {
		vol->pix_type = PT_FLOAT;
		vol->pix_size = sizeof(float);
	    }
	    continue;
	}
	if (strcmp (linebuf, "ElementType = MET_SHORT") == 0) {
	    vol->pix_type = PT_SHORT;
	    vol->pix_size = sizeof(short);
	    continue;
	}
	if (strcmp (linebuf, "ElementType = MET_UCHAR") == 0) {
	    vol->pix_type = PT_UCHAR;
	    vol->pix_size = sizeof(unsigned char);
	    continue;
	}
    }

    if (vol->pix_size <= 0) {
	printf ("Oops, couldn't interpret mha data type\n");
	exit (-1);
    }
        
    if (mh5) {
	fseek(fp, 512, SEEK_SET);
    }

    vol->img = malloc (vol->pix_size*vol->npix);
    if (!vol->img) {
	printf ("Oops, out of memory\n");
	exit (-1);
    }

    rc = fread (vol->img, vol->pix_size, vol->npix, fp);
    if (rc != vol->npix) {
	printf ("Oops, bad read from file (%d)\n", rc);
	exit (-1);
    }
    fclose (fp);

    return vol;
}

/* Return 1 if filename ends in ".mh5" */
static int 
is_mh5 (const char* filename)
{
    int len = strlen (filename);
    if (len < 4) return 0;
    if (!strcmp (&filename[len-4], ".mh5")) return 1;
    if (!strcmp (&filename[len-4], ".MH5")) return 1;
    return 0;
}


/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
void 
write_mha (const char* filename, Volume* vol)
{
    if (is_mh5 (filename)) {
	write_mha_internal (filename, vol, 1);
    } else {
	write_mha_internal (filename, vol, 0);
    }
}

Volume* 
read_mha (const char* filename)
{
    if (is_mh5 (filename)) {
	return read_mha_internal (filename, 1);
    } else {
	return read_mha_internal (filename, 0);
    }
    

}
