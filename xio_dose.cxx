/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>

#include <itksys/SystemTools.hxx>
#include <itksys/Directory.hxx>
#include <itksys/RegularExpression.hxx>
#include "itkDirectory.h"
#include "itkRegularExpressionSeriesFileNames.h"
#include "bstrlib.h"
#include "gdcmFile.h"
#include "gdcmFileHelper.h"
#include "gdcmGlobal.h"
#include "gdcmSeqEntry.h"
#include "gdcmSQItem.h"
#include "gdcmUtil.h"

#include "cxt.h"
#include "gdcm_series.h"
#include "plm_image.h"
#include "plm_image_type.h"
#include "plm_image_patient_position.h"
#include "print_and_exit.h"
#include "xio_ct.h"
#include "xio_dose.h"
#include "xio_io.h"
#include "xio_structures.h"

#define XIO_DATATYPE_UINT32 5

typedef struct xio_dose_header Xio_dose_header;
struct xio_dose_header {
    int dim[3];
    float offset[3];
    float spacing[3];
    double dose_scale_factor;
    double dose_weight;
};

static void
xio_dose_load_header (Xio_dose_header *xdh, const char *filename)
{
    FILE *fp;
    struct bStream * bs;
    bstring line1 = bfromcstr ("");
    int i;
    int rc;
    int dummy;
    /* Header info */
    Xio_version xio_dose_version;
    int xio_dose_datatype;
    int xio_sources;
    double xio_dose_scalefactor, xio_dose_weight;
    /* Dose cube definition */
    double rx; double ry; double rz;
    double ox; double oy; double oz;
    int nx; int ny; int nz;
    /* Element spacing */
    double dx; double dy; double dz;
    /* Offset */
    double topx; double topy; double topz;

    fp = fopen (filename, "r");
    if (!fp) {
	print_and_exit ("Error opening file %s for read\n", filename);
    }

    bs = bsopen ((bNread) fread, fp);

    /* XiO file format version */
    bsreadln (line1, bs, '\n');

    if (!strncmp ((char*) line1->data, "006d101e", strlen ("006d101e"))) {
        xio_dose_version = XIO_VERSION_4_5_0;
    } else if (!strncmp ((char*) line1->data, "0062101e", strlen ("0062101e"))) {
        xio_dose_version = XIO_VERSION_4_33_02;
    } else if (!strncmp ((char*) line1->data, "004f101e", strlen ("004f101e"))) {
        xio_dose_version = XIO_VERSION_4_2_1;
    } else {
        xio_dose_version = XIO_VERSION_UNKNOWN;
    }

    if (xio_dose_version == XIO_VERSION_UNKNOWN) {
        print_and_exit ("Error. Unknown XiO file format version: %s\n", line1->data);
    }

    /* Skip line */
    bsreadln (line1, bs, '\n');

    if (xio_dose_version == XIO_VERSION_4_33_02 || xio_dose_version == XIO_VERSION_4_5_0)
    {
	/* Skip line */
	bsreadln (line1, bs, '\n');
    }

    /* Number of subplans or beams */
    bsreadln (line1, bs, '\n');
    rc = sscanf ((char*) line1->data, "%1d", &xio_sources);

    if (rc != 1) {
        print_and_exit ("Error. Cannot parse sources/subplans: %s\n", line1->data);
    }

    printf ("Dose file is a sum of %d sources/subplans:\n", xio_sources);

    /* One line for each source/subplan */
    for (i = 1; i <= xio_sources; i++) {
        bsreadln (line1, bs, '\n');
        printf ("Source/subplan %d: %s", i, line1->data);
    }

    /* Dose normalization info */
    bsreadln (line1, bs, '\n');
    rc = sscanf ((char*) line1->data, "%lf,%lf", &xio_dose_scalefactor, &xio_dose_weight);

    if (rc != 2) {
        print_and_exit ("Error. Cannot parse dose normalization: %s\n", line1->data);
    }

    printf ("Dose scale factor = %f\n", xio_dose_scalefactor);
    printf ("Dose weight = %f\n", xio_dose_weight);

    /* Skip line */
    bsreadln (line1, bs, '\n');

    /* Data type */
    bsreadln (line1, bs, '\n');
    rc = sscanf ((char*) line1->data, "%1d", &xio_dose_datatype);

    if (rc != 1) {
        print_and_exit ("Error. Cannot parse datatype: %s\n", line1->data);
    }

    if (xio_dose_datatype != XIO_DATATYPE_UINT32) {
        print_and_exit ("Error. Only unsigned 32-bit integer data is currently supported: %s\n", line1->data);
    }

    /* Dose cube definition */
    bsreadln (line1, bs, '\n');

    rc = sscanf ((char*) line1->data, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%d,%d,%d",
                 &dummy, &rx, &rz, &ry, &ox, &oz, &oy, &nx, &nz, &ny);

    if (rc != 10) {
        print_and_exit ("Error. Cannot parse dose cube definition: %s\n", line1->data);
    }

    printf ("rx = %lf, ry = %lf, rz = %lf\n", rx, ry, rz);
    printf ("ox = %lf, oy = %lf, oz = %lf\n", ox, oy, oz);
    printf ("nx = %d, ny = %d, nz = %d\n", nx, ny, nz);

    /* Calculate element spacing */
    dx = rx / (nx - 1);
    dy = ry / (ny - 1);
    dz = rz / (nz - 1);

    /* Calculate offset */
    topx = ox - (rx / 2);
    topy = oy - (ry / 2);
    topz = -oz - (rz / 2);

    /* Put info into header */
    xdh->dim[0] = nx;
    xdh->dim[1] = nz;
    xdh->dim[2] = ny;

    xdh->spacing[0] = dx;
    xdh->spacing[1] = dz;
    xdh->spacing[2] = dy;

    xdh->offset[0] = topx;
    xdh->offset[1] = topz;
    xdh->offset[2] = topy;

    xdh->dose_scale_factor = xio_dose_scalefactor;
    xdh->dose_weight = xio_dose_weight;

    /* Clean up */
    bdestroy (line1);
    bsclose (bs);
    fclose (fp);
}

static void
xio_dose_load_cube (
    Plm_image *pli,
    Xio_dose_header *xdh,
    const char *filename
)
{
    FILE *fp;
    Volume *v;
    unsigned int *cube_img_read;
    float *cube_img_normalize;
    int i, rc;

    v = (Volume*) pli->m_gpuit;
    cube_img_read = (unsigned int*) v->img;

    fp = fopen (filename, "rb");
    if (!fp) {
	print_and_exit ("Error opening file %s for read\n", filename);
    }

    /* Read dose cube */
    rc = fseek (fp, - v->dim[0] * v->dim[1] * v->dim[2] * sizeof(unsigned int), SEEK_END);
    if (rc == -1) {
	print_and_exit ("Error seeking backward when reading image file\n");
    }
    rc = fread (cube_img_read, sizeof(unsigned int), v->dim[0] * v->dim[1] * v->dim[2], fp);
    if (rc != v->dim[0] * v->dim[1] * v->dim[2]) {
	perror ("File error: ");
	print_and_exit (
	    "Error reading xio dose cube (%s)\n"
	    "  rc = %d, ferror = %d\n", 
	    filename, rc, ferror (fp));
    }

    /* Switch big-endian to little-endian */
    for (i = 0; i < v->dim[0] * v->dim[1] * v->dim[2]; i++) {
	char lenbuf[4];
	char tmpc;
	memcpy (lenbuf, (char*) &cube_img_read[i], 4);
	tmpc = lenbuf[0]; lenbuf[0] = lenbuf[3]; lenbuf[3] = tmpc;
	tmpc = lenbuf[1]; lenbuf[1] = lenbuf[2]; lenbuf[2] = tmpc;
	memcpy ((char*) &cube_img_read[i], lenbuf, 4);
    }

    /* Convert volume to float for more accurate normalization */
    pli->convert (PLM_IMG_TYPE_GPUIT_FLOAT);
    v = (Volume*) pli->m_gpuit;
    cube_img_normalize = (float*) v->img;

    /* Normalize dose */
    volume_scale (v, xdh->dose_weight * xdh->dose_scale_factor);

    fclose (fp);
}

static void
xio_dose_create_volume (
    Plm_image *pli, 
    Xio_dose_header *xdh
)
{
    Volume *v;

    v = volume_create (xdh->dim, xdh->offset, xdh->spacing, PT_UINT32, 0, 0);
    pli->set_gpuit (v);

    printf ("img: %p\n", v->img);
    printf ("Image dim: %d %d %d\n", v->dim[0], v->dim[1], v->dim[2]);
}

void
xio_dose_load (Plm_image *pli, const char *filename)
{
    Xio_dose_header xdh;
    
    xio_dose_load_header(&xdh, filename);
    xio_dose_create_volume(pli, &xdh);
    xio_dose_load_cube(pli, &xdh, filename);
}

void
xio_dose_apply_transform (Plm_image *pli, Xio_ct_transform *transform)
{
    int i;

    Volume *v;
    v = (Volume*) pli->m_gpuit;

    /* Set patient position */
    pli->m_patient_pos = transform->patient_pos;

    /* Set offsets */
    v->offset[0] = (v->offset[0] * transform->direction_cosines[0]) + transform->x_offset;
    v->offset[1] = (v->offset[1] * transform->direction_cosines[4]) + transform->y_offset;

    /* Set direction cosines */
    for (i = 0; i <= 8; i++) {
    	v->direction_cosines[i] = transform->direction_cosines[i];
    }
}
