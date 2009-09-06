/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if (defined(_WIN32) || defined(WIN32))
#include <io.h>        // windows //
#endif
#include "plm_config.h"
#include "volume.h"

#define LINELEN 128
#define MIN_SHORT -32768
#define MAX_SHORT 32767

#define WRITE_BLOCK (1024*1024)

/* GCS Jun 18, 2008.  When using MSVC 2005, large fwrite calls to files over 
    samba mount fails.  This seems to be a bug in the C runtime library.
    This function works around the problem by breaking up the large write 
    into many "medium-sized" writes. */
void fwrite_block (void* buf, size_t size, size_t count, FILE* fp)
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
	    fprintf (stderr, "Error writing to file.  rc=%d, this_write=%d\n",
		    rc, this_write);
	    return;
	}
	cur += rc;
	left_to_write -= rc;
    }
}

gpuit_EXPORT
void write_mha (char* filename, Volume* vol)
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
    case PT_ULONG:
	element_type = "MET_ULONG";
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
    fprintf (fp, mha_header, 
	     vol->offset[0], vol->offset[1], vol->offset[2], 
	     vol->pix_spacing[0], vol->pix_spacing[1], vol->pix_spacing[2], 
	     vol->dim[0], vol->dim[1], vol->dim[2],
	     (vol->pix_type == PT_VF_FLOAT_INTERLEAVED) 
		? "ElementNumberOfChannels = 3\n" : "",
	     element_type);
    fflush (fp);

    fwrite_block (vol->img, vol->pix_size, vol->npix, fp);

    fclose (fp);
}

gpuit_EXPORT
Volume* read_mha (char* filename)
{
    int rc;
    char linebuf[LINELEN];
    Volume* vol = (Volume*) malloc (sizeof(Volume));
    int tmp;
    FILE* fp;


    fp = fopen (filename,"rb");
    if (!fp) {
	fprintf (stderr, "File %s not found\n", filename);
	return 0;
    }
    
    vol->pix_size = -1;
    vol->pix_type = PT_UNDEFINED;
    while (fgets(linebuf,LINELEN,fp)) {
	if (strcmp (linebuf, "ElementDataFile = LOCAL\n") == 0) {
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
	if (sscanf (linebuf, "ElementNumberOfChannels = %d", &tmp) == 1) {
	    if (vol->pix_type == PT_UNDEFINED || vol->pix_type == PT_FLOAT) {
		vol->pix_type = PT_VF_FLOAT_INTERLEAVED;
		vol->pix_size = 3*sizeof(float);
	    }
	    continue;
	}
	if (strcmp (linebuf, "ElementType = MET_FLOAT\n") == 0) {
	    if (vol->pix_type == PT_UNDEFINED) {
		vol->pix_type = PT_FLOAT;
		vol->pix_size = sizeof(float);
	    }
	    continue;
	}
	if (strcmp (linebuf, "ElementType = MET_SHORT\n") == 0) {
	    vol->pix_type = PT_SHORT;
	    vol->pix_size = sizeof(short);
	    continue;
	}
	if (strcmp (linebuf, "ElementType = MET_UCHAR\n") == 0) {
	    vol->pix_type = PT_UCHAR;
	    vol->pix_size = sizeof(unsigned char);
	    continue;
	}
    }
    if (vol->pix_size <= 0) {
	printf ("Oops, couldn't interpret mha data type\n");
	exit (-1);
    }
    printf ("pixsize = %d, npix = %d\n", vol->pix_size, vol->npix);
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
    printf ("Read OK!\n");
    fclose (fp);

#if defined (commentout)
    /* Compute some auxiliary variables */
    vol->xmin = vol->offset[0] - vol->pix_spacing[0] / 2;
    vol->xmax = vol->xmin + vol->pix_spacing[0] * vol->dim[0];
    vol->ymin = vol->offset[1] - vol->pix_spacing[1] / 2;
    vol->ymax = vol->ymin + vol->pix_spacing[1] * vol->dim[1];
    vol->zmin = vol->offset[2] - vol->pix_spacing[2] / 2;
    vol->zmax = vol->zmin + vol->pix_spacing[2] * vol->dim[2];
#endif

    return vol;
}
