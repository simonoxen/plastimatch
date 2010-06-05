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
#include "xio_io.h"
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

    /* Skip 9 lines */
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
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

    /* Skip 7 lines */
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
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

    /* Skip 1 lines */
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
    short *img, *slice_img;;
    int i, rc;

    v = (Volume*) pli->m_gpuit;
    img = (short*) v->img;
    slice_img = &img[slice_no * v->dim[0] * v->dim[1]];

    fp = fopen (filename, "rb");
    if (!fp) {
	print_and_exit ("Error opening file %s for read\n", filename);
    }

    /* Load image */
    rc = fseek (fp, - v->dim[0] * v->dim[1] * sizeof(short), SEEK_END);
    if (rc == -1) {
	print_and_exit ("Error seeking backward when reading image file\n");
    }
    rc = fread (slice_img, sizeof(short), v->dim[0] * v->dim[1], fp);
    if (rc != v->dim[0] * v->dim[1]) {
	perror ("File error: ");
	print_and_exit (
	    "Error reading xio ct image (%s)\n"
	    "  rc = %d, ferror = %d\n", 
	    filename, rc, ferror (fp));
    }

    /* Switch big-endian to little-endian */
    for (i = 0; i < v->dim[0] * v->dim[1]; i++) {
	char *byte = (char*) &slice_img[i];
	char tmp = byte[0];
	byte[0] = byte[1];
	byte[1] = tmp;
    }

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
    int dim[3];
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

    v = volume_create (dim, offset, spacing, PT_SHORT, 0, 0);
    pli->set_gpuit (v);

    printf ("img: %p\n", v->img);
    printf ("Image dim: %d %d %d\n", v->dim[0], v->dim[1], v->dim[2]);
}

void
xio_ct_load (Plm_image *pli, char *input_dir)
{
    //const char *filename_re = "T\\.([-\\.0-9]*)\\.(CT|MR)";
    const char *filename_re = "T\\.([-\\.0-9]*)\\.(CT|MR)";
    const char *filename;
    std::vector<std::pair<std::string,std::string> >::iterator it;
    int i;
    Xio_ct_header xch;
    float z_prev, z_diff;
    int this_chunk_len, best_chunk_len;
    int this_chunk_start, best_chunk_start;
    float this_chunk_diff = 0.0f, best_chunk_diff = 0.0f;

    /* Get the list of filenames */
    std::vector<std::pair<std::string,std::string> > file_names;
    xio_io_get_file_names (&file_names, input_dir, filename_re);
    if (file_names.empty ()) {
	print_and_exit ("No xio CT files found in directory %s\n", 
			input_dir);
    }

    /* Iterate through filenames, find largest chunk */
    it = file_names.begin();
    filename = (*it).first.c_str();
    xio_ct_load_header (&xch, filename);
    z_prev = xch.z_loc;
    this_chunk_start = best_chunk_start = 0;
    this_chunk_len = best_chunk_len = 0;
    ++it;
    i = 1;
    while (it != file_names.end()) {
	filename = (*it).first.c_str();
	xio_ct_load_header (&xch, filename);
	z_diff = xch.z_loc - z_prev;
	printf ("%f, %f\n", xch.z_loc, z_diff);
	z_prev = xch.z_loc;
	if (best_chunk_len == 0) {
	    this_chunk_start = i - 1;
	    this_chunk_diff = best_chunk_diff = z_diff;
	    this_chunk_len = best_chunk_len = 2;
	} else if (fabs (this_chunk_diff - z_diff) < 0.01) {
	    this_chunk_len ++;
	} else {
	    printf ("RESET CHUNK: %d\n", i);
	    this_chunk_start = i - 1;
	    this_chunk_len = 2;
	    this_chunk_diff = z_diff;
	}
	if (this_chunk_len > best_chunk_len) {
	    best_chunk_start = this_chunk_start;
	    best_chunk_len = this_chunk_len;
	    best_chunk_diff = this_chunk_diff;
	}
	++it;
	++i;
    }

    printf ("Found best chunk: %d, %d, %g\n", best_chunk_start, 
	best_chunk_len, best_chunk_diff);

    /* Iterate through filenames, adding data to pli */
    i = 0;
    it = file_names.begin();
    while (it != file_names.end()) {
	filename = (*it).first.c_str();
	if (i == best_chunk_start) {
	    xio_ct_load_header (&xch, filename);
	    xio_ct_create_volume (pli, &xch, best_chunk_len, best_chunk_diff);
	}
	if (i >= best_chunk_start && i < best_chunk_start + best_chunk_len ) {
	    xio_ct_load_image (pli, i - best_chunk_start, filename);
	}
	++i;
	++it;
    }
}

void
xio_ct_get_transform_from_dicom_dir (
    Plm_image *pli, 
    Xio_ct_transform *transform, 
    char *dicom_dir
)
{
    /* Use original XiO CT geometry and a DICOM directory to determine
       the transformation from XiO CT coordinates to DICOM coordinates.
       Because structures and dose in XiO are defined relative to CT
       geometry, the transform can be applied to those as well. */

    int i;

    Volume *v;
    v = (Volume*) pli->m_gpuit;

    Gdcm_series gs;
    std::string tmp;
    
    if (!dicom_dir) {
	return;
    }

    gs.load (dicom_dir);
    gs.digest_files ();
    if (!gs.m_have_ct) {
	return;
    }

    /* Initialize */
    for (i = 0; i <= 8; i++) {
	transform->direction_cosines[i] = 0.0f;
    }

    if ( (pli->m_patient_pos == PATIENT_POSITION_HFS) ||
	(pli->m_patient_pos == PATIENT_POSITION_UNKNOWN) ) {

	/* Offsets */
	transform->x_offset = v->offset[0] - gs.m_origin[0];
	transform->y_offset = v->offset[1] - gs.m_origin[1];

	/* Direction cosines */
	transform->direction_cosines[0] = 1.0f;
	transform->direction_cosines[4] = 1.0f;
	transform->direction_cosines[8] = 1.0f;

    } else if (pli->m_patient_pos == PATIENT_POSITION_HFP) {

	/* Offsets */
	transform->x_offset = v->offset[0] + gs.m_origin[0];
	transform->y_offset = v->offset[1] + gs.m_origin[1];

	/* Direction cosines */
	transform->direction_cosines[0] = -1.0f;
	transform->direction_cosines[4] = -1.0f;
	transform->direction_cosines[8] = 1.0f;

    } else if (pli->m_patient_pos == PATIENT_POSITION_FFS) {

	/* Offsets */
	transform->x_offset = v->offset[0] + gs.m_origin[0];
	transform->y_offset = v->offset[1] - gs.m_origin[1];

	/* Direction cosines */
	transform->direction_cosines[0] = -1.0f;
	transform->direction_cosines[4] = 1.0f;
	transform->direction_cosines[8] = -1.0f;

    } else if (pli->m_patient_pos == PATIENT_POSITION_FFP) {

	/* Offsets */
	transform->x_offset = v->offset[0] - gs.m_origin[0];
	transform->y_offset = v->offset[1] + gs.m_origin[1];

	/* Direction cosines */
	transform->direction_cosines[0] = 1.0f;
	transform->direction_cosines[4] = -1.0f;
	transform->direction_cosines[8] = -1.0f;
    }
}

void
xio_ct_get_transform (
    Plm_image *pli, 
    Xio_ct_transform *transform
)
{
    /* Use patient position to determine the transformation from XiO CT
       coordinates to DICOM coordinates. Only the direction cosines
       are modified, the original will remain the same.
       Because structures and dose in XiO are defined relative to CT
       geometry, the transform can be applied to those as well. */

    if ( (pli->m_patient_pos == PATIENT_POSITION_HFS) ||
	(pli->m_patient_pos == PATIENT_POSITION_UNKNOWN) ) {

	transform->direction_cosines[0] = 1.0f;
	transform->direction_cosines[4] = 1.0f;
	transform->direction_cosines[8] = 1.0f;

    } else if (pli->m_patient_pos == PATIENT_POSITION_HFP) {

	transform->direction_cosines[0] = -1.0f;
	transform->direction_cosines[4] = -1.0f;
	transform->direction_cosines[8] = 1.0f;

    } else if (pli->m_patient_pos == PATIENT_POSITION_FFS) {

	transform->direction_cosines[0] = -1.0f;
	transform->direction_cosines[4] = 1.0f;
	transform->direction_cosines[8] = -1.0f;

    } else if (pli->m_patient_pos == PATIENT_POSITION_FFP) {

	transform->direction_cosines[0] = 1.0f;
	transform->direction_cosines[4] = -1.0f;
	transform->direction_cosines[8] = -1.0f;

    }

}

void
xio_ct_apply_transform (Plm_image *pli, Xio_ct_transform *transform)
{
    int i;

    Volume *v;
    v = (Volume*) pli->m_gpuit;

    /* Set offsets */
    v->offset[0] = (v->offset[0] * transform->direction_cosines[0]) + transform->x_offset;
    v->offset[1] = (v->offset[1] * transform->direction_cosines[4]) + transform->y_offset;

    /* Set direction cosines */
    for (i = 0; i <= 8; i++) {
    	v->direction_cosines[i] = transform->direction_cosines[i];
    }
}
