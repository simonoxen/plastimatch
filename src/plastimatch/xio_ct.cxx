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

#if GDCM_VERSION_1
#include "gdcm1_series.h"
#endif
#include "plm_endian.h"
#include "plm_image.h"
#include "plm_image_type.h"
#include "print_and_exit.h"
#include "referenced_dicom_dir.h"
#include "rtss_polyline_set.h"
#include "xio_ct.h"
#include "xio_studyset.h"
#include "xio_structures.h"

typedef struct xio_ct_header Xio_ct_header;
struct xio_ct_header {
    float slice_size[2];
    int dim[2];
    int bit_depth;
    float spacing[2];
    float z_loc;
};

static void
xio_ct_load_header (Xio_ct_header *xch, const char *filename)
{
    FILE *fp;
    struct bStream * bs;
    bstring line1 = bfromcstr ("");
    int rc;

    fp = fopen (filename, "r");
    if (!fp) {
	print_and_exit ("Error opening file %s for read\n", filename);
    }

    bs = bsopen ((bNread) fread, fp);

    /* Skip 5 lines */
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');

    /* Get slice location */
    rc = bsreadln (line1, bs, '\n');
    if (rc == BSTR_ERR) {
	print_and_exit ("Error reading slice location\n", line1->data);
    }
    rc = sscanf ((char*) line1->data, "%g", &xch->z_loc);
    if (rc != 1) {
	print_and_exit ("Error parsing slice location (%s)\n", line1->data);
    }

    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');

    /* Get slice width, height */
    rc = bsreadln (line1, bs, '\n');
    if (rc == BSTR_ERR) {
	print_and_exit ("Error reading image resolution\n", line1->data);
    }
    rc = sscanf ((char*) line1->data, "%g,%g", &xch->slice_size[0], 
	&xch->slice_size[1]);
    if (rc != 2) {
	print_and_exit ("Error parsing slice width (%s)\n", line1->data);
    }

    /* Get image resolution */
    rc = bsreadln (line1, bs, '\n');
    if (rc == BSTR_ERR) {
	print_and_exit ("Error reading image resolution\n", line1->data);
    }
    rc = sscanf ((char*) line1->data, "%d,%d,%d", &xch->dim[0], &xch->dim[1], 
	&xch->bit_depth);
    if (rc != 3) {
	print_and_exit ("Error parsing image resolution (%s)\n", line1->data);
    }

    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');

    /* Get pixel size */
    rc = bsreadln (line1, bs, '\n');
    if (rc == BSTR_ERR) {
	print_and_exit ("Error reading pixel size\n", line1->data);
    }
    rc = sscanf ((char*) line1->data, "%g,%g", &xch->spacing[0], 
	&xch->spacing[1]);
    if (rc != 2) {
	print_and_exit ("Error parsing pixel size (%s)\n", line1->data);
    }

    printf ("%g %g %d %d %g %g %g\n", 
	xch->slice_size[0], xch->slice_size[1], xch->dim[0], xch->dim[1], 
	xch->spacing[0], xch->spacing[1], xch->z_loc);

    bdestroy (line1);
    bsclose (bs);
    fclose (fp);
}

static void
xio_ct_load_image (
    Plm_image *pli, 
    int slice_no,
    const char *filename
)
{
    FILE *fp;
    Volume *v;
    short *img, *slice_img;
    int rc1;
    size_t rc2;

    v = (Volume*) pli->m_gpuit;
    img = (short*) v->img;
    slice_img = &img[slice_no * v->dim[0] * v->dim[1]];

    fp = fopen (filename, "rb");
    if (!fp) {
	print_and_exit ("Error opening file %s for read\n", filename);
    }

    /* Load image */
    rc1 = fseek (fp, - v->dim[0] * v->dim[1] * sizeof(short), SEEK_END);
    if (rc1 == -1) {
	print_and_exit ("Error seeking backward when reading image file\n");
    }
    rc2 = fread (slice_img, sizeof(short), v->dim[0] * v->dim[1], fp);
    if (rc2 != v->dim[0] * v->dim[1]) {
	perror ("File error: ");
	print_and_exit (
	    "Error reading xio ct image (%s)\n"
	    "  rc = %u, ferror = %d\n", 
	    filename, (unsigned int) rc2, ferror (fp));
    }

    /* Switch big-endian to native */
#if defined (commentout)
    for (int i = 0; i < v->dim[0] * v->dim[1]; i++) {
	char *byte = (char*) &slice_img[i];
	char tmp = byte[0];
	byte[0] = byte[1];
	byte[1] = tmp;
    }
#endif
    endian2_big_to_native ((void*) slice_img, v->dim[0] * v->dim[1]);

    fclose (fp);
}

static void
xio_ct_create_volume (
    Plm_image *pli, 
    Xio_ct_header *xch,
    int best_chunk_len,
    float best_chunk_diff
)
{
    Volume *v;
    size_t dim[3];
    float offset[3];
    float spacing[3];

    dim[0] = xch->dim[0];
    dim[1] = xch->dim[1];
    dim[2] = best_chunk_len;

    offset[0] = - xch->slice_size[0];
    offset[1] = - xch->slice_size[1];
    offset[2] = xch->z_loc;

    spacing[0] = xch->spacing[0];
    spacing[1] = xch->spacing[1];
    spacing[2] = best_chunk_diff;

    v = new Volume (dim, offset, spacing, 0, PT_SHORT, 1);
    pli->set_gpuit (v);

    printf ("img: %p\n", v->img);
    printf ("Image dim: %u %u %u\n", (unsigned int) v->dim[0], 
	(unsigned int) v->dim[1], (unsigned int) v->dim[2]);
}

void
xio_ct_load (Plm_image *pli, const Xio_studyset *studyset)
{
    int i;

    Xio_ct_header xch;
    string ct_file;

    if (studyset->number_slices > 0) {
	ct_file = studyset->studyset_dir 
	    + "/" + studyset->slices[0].filename_scan.c_str();
        xio_ct_load_header (&xch, ct_file.c_str());
	xio_ct_create_volume (pli, &xch, studyset->number_slices, 
	    studyset->thickness);

	for (i = 0; i < studyset->number_slices; i++) {
	    ct_file = studyset->studyset_dir 
		+ "/" + studyset->slices[i].filename_scan.c_str();
	    xio_ct_load_image (pli, i, ct_file.c_str());
	}
    }
}

void
xio_ct_get_transform_from_rdd (
    Plm_image *pli,
    Img_metadata *img_metadata,
    Referenced_dicom_dir *rdd,
    Xio_ct_transform *transform
)
{
    /* Use original XiO CT geometry and a DICOM directory to determine
       the transformation from XiO coordinates to DICOM coordinates. */

    Volume *v;
    v = (Volume*) pli->m_gpuit;

    /* Offsets */
    transform->x_offset = 0;
    transform->y_offset = 0;

    /* Direction cosines */
    for (int i = 0; i <= 8; i++) {
	transform->direction_cosines[i] = 0.;
    }
    transform->direction_cosines[0] = 1.0f;
    transform->direction_cosines[4] = 1.0f;
    transform->direction_cosines[8] = 1.0f;

    std::string patient_pos = img_metadata->get_metadata(0x0018, 0x5100);

    if (patient_pos == "HFS" ||	patient_pos == "") {

	/* Offsets */
	transform->x_offset = v->offset[0] - rdd->m_pih.m_origin[0];
	transform->y_offset = v->offset[1] - rdd->m_pih.m_origin[1];

	/* Direction cosines */
	transform->direction_cosines[0] = 1.0f;
	transform->direction_cosines[4] = 1.0f;
	transform->direction_cosines[8] = 1.0f;

    } else if (patient_pos == "HFP") {

	/* Offsets */
	transform->x_offset = v->offset[0] + rdd->m_pih.m_origin[0];
	transform->y_offset = v->offset[1] + rdd->m_pih.m_origin[1];

	/* Direction cosines */
	transform->direction_cosines[0] = -1.0f;
	transform->direction_cosines[4] = -1.0f;
	transform->direction_cosines[8] = 1.0f;

    } else if (patient_pos == "FFS") {

	/* Offsets */
	transform->x_offset = v->offset[0] + rdd->m_pih.m_origin[0];
	transform->y_offset = v->offset[1] - rdd->m_pih.m_origin[1];

	/* Direction cosines */
	transform->direction_cosines[0] = -1.0f;
	transform->direction_cosines[4] = 1.0f;
	transform->direction_cosines[8] = -1.0f;

    } else if (patient_pos == "FFP") {

	/* Offsets */
	transform->x_offset = v->offset[0] - rdd->m_pih.m_origin[0];
	transform->y_offset = v->offset[1] + rdd->m_pih.m_origin[1];

	/* Direction cosines */
	transform->direction_cosines[0] = 1.0f;
	transform->direction_cosines[4] = -1.0f;
	transform->direction_cosines[8] = -1.0f;
    }

}

void
xio_ct_get_transform (
    Img_metadata *img_metadata,
    Xio_ct_transform *transform
)
{
    /* Use patient position to determine the transformation from XiO
       coordinates to a valid set of DICOM coordinates.
       The origin of the DICOM coordinates will be the same as the
       origin of the XiO coordinates, which is generally not the same
       as the origin of the original DICOM CT scan. */

    /* Offsets */
    transform->x_offset = 0;
    transform->y_offset = 0;

    /* Direction cosines */
    for (int i = 0; i <= 8; i++) {
	transform->direction_cosines[i] = 0.;
    }
    transform->direction_cosines[0] = 1.0f;
    transform->direction_cosines[4] = 1.0f;
    transform->direction_cosines[8] = 1.0f;

    std::string patient_pos = img_metadata->get_metadata(0x0018, 0x5100);

    if (patient_pos == "HFS" ||	patient_pos == "") {

	transform->direction_cosines[0] = 1.0f;
	transform->direction_cosines[4] = 1.0f;
	transform->direction_cosines[8] = 1.0f;

    } else if (patient_pos == "HFP") {

	transform->direction_cosines[0] = -1.0f;
	transform->direction_cosines[4] = -1.0f;
	transform->direction_cosines[8] = 1.0f;

    } else if (patient_pos == "FFS") {

	transform->direction_cosines[0] = -1.0f;
	transform->direction_cosines[4] = 1.0f;
	transform->direction_cosines[8] = -1.0f;

    } else if (patient_pos == "FFP") {

	transform->direction_cosines[0] = 1.0f;
	transform->direction_cosines[4] = -1.0f;
	transform->direction_cosines[8] = -1.0f;

    }

}

void
xio_ct_apply_transform (Plm_image *pli, Xio_ct_transform *transform)
{
    /* Transform coordinates of an XiO CT scan to DICOM coordinates */

    Volume *v;
    v = (Volume*) pli->m_gpuit;

    /* Set offsets */
    v->offset[0] = (v->offset[0] * transform->direction_cosines[0]) + transform->x_offset;
    v->offset[1] = (v->offset[1] * transform->direction_cosines[4]) + transform->y_offset;

    /* Set direction cosines */
    v->set_direction_cosines (transform->direction_cosines);
}
