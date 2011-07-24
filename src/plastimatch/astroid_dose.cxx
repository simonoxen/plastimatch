/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <string.h>

#include "plm_image.h"
#include "plm_image_type.h"
#include "plm_image_patient_position.h"
#include "plm_int.h"
#include "print_and_exit.h"
#include "xio_ct.h"

typedef struct astroid_dose_header Astroid_dose_header;
struct astroid_dose_header {
    int dim[3];
    float offset[3];
    float spacing[3];
};

static void
astroid_dose_load_header (Astroid_dose_header *adh, const char *filename)
{    
    FILE *fp;

    int rc;

    /* Dose cube definition */
    double rx; double ry; double rz;
    double ox; double oy; double oz;
    int nx; int ny; int nz;
    /* Element spacing */
    double dx; double dy; double dz;
    /* Offset */
    double topx; double topy; double topz;

    char line1[1024];

    /* Astroid doesn't include the geometry in the dose export file.
       Therefore an additional <filename>.geometry file will be loaded which
       should contains a single line that defines the geometry with 9 values
       in XiO coordinates:

       rx, rz, ry, ox, oz, oy, nx, nz, ny

       Coordinates in Astroid are the same as in the imported XiO plans.
       To transform to DICOM coordinates, both the XiO studyset and original CT
       need to be loaded:

       --input-dose-ast <astroid-dose>
       --input <xio-studyset-dir>
       --dicom-dir <original-ct>
    */

    std::string filename_geometry = std::string(filename) + ".geometry";

    fp = fopen (filename_geometry.c_str(), "rb");
    if (!fp) {
	print_and_exit ("Error opening geometry file %s for read\n",
	    filename_geometry.c_str());
    }

    /* Dose cube definition */
    fgets (line1, sizeof(line1), fp);

    rc = sscanf (line1, "%lf,%lf,%lf,%lf,%lf,%lf,%d,%d,%d",
                 &rx, &rz, &ry, &ox, &oz, &oy, &nx, &nz, &ny);

    if (rc != 9) {
        print_and_exit ("Error. Cannot parse dose cube definition: %s\n", line1);
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
    adh->dim[0] = nx;
    adh->dim[1] = nz;
    adh->dim[2] = ny;

    adh->spacing[0] = dx;
    adh->spacing[1] = dz;
    adh->spacing[2] = dy;

    adh->offset[0] = topx;
    adh->offset[1] = topz;
    adh->offset[2] = topy;

    fclose (fp);
}

static void
astroid_dose_load_cube (
    Plm_image *pli,
    Astroid_dose_header *adh,
    const char *filename
)
{
    FILE *fp;
    Volume *v;
    uint32_t *cube_img_read;
    int i, j, k, rc;

    v = (Volume*) pli->m_gpuit;
    cube_img_read = (uint32_t*) v->img;

    fp = fopen (filename, "rb");
    if (!fp) {
	print_and_exit ("Error opening file %s for read\n", filename);
    }

    /* Read dose cube */
    rc = fread (cube_img_read, sizeof(uint32_t), v->dim[0] * v->dim[1] * v->dim[2], fp);
    if (rc != v->dim[0] * v->dim[1] * v->dim[2]) {
	perror ("File error: ");
	print_and_exit (
	    "Error reading astroid dose cube (%s)\n"
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

    /* Flip XiO Z axis */
    Volume* vflip;
    vflip = new Volume (v->dim, v->offset, v->spacing, 
	v->direction_cosines, v->pix_type, v->vox_planes, 0);

    for (k=0;k<v->dim[2];k++) {
	for (j=0;j<v->dim[1];j++) {
	    for (i=0;i<v->dim[0];i++) {
		memcpy ((float*)vflip->img
		    + volume_index (vflip->dim, i, (vflip->dim[1]-1-j), k), 
		    (float*)v->img 
		    + volume_index (v->dim, i, j, k), v->pix_size);
	    }
	}
    }

    pli->set_gpuit (vflip);

    /* Convert volume to float for more accurate normalization */
    pli->convert (PLM_IMG_TYPE_GPUIT_FLOAT);

    /* Convert from cGy to Gy */
    volume_scale (vflip, 0.01);

    fclose (fp);
}

static void
astroid_dose_create_volume (
    Plm_image *pli, 
    Astroid_dose_header *adh
)
{
    Volume *v;

    v = new Volume (adh->dim, adh->offset, adh->spacing, 0, 
	PT_UINT32, 1, 0);
    pli->set_gpuit (v);

    printf ("img: %p\n", v->img);
    printf ("Image dim: %d %d %d\n", v->dim[0], v->dim[1], v->dim[2]);
}

void
astroid_dose_load (Plm_image *pli, const char *filename)
{
    Astroid_dose_header adh;
    
    astroid_dose_load_header(&adh, filename);
    astroid_dose_create_volume(pli, &adh);
    astroid_dose_load_cube(pli, &adh, filename);
}

void
astroid_dose_apply_transform (Plm_image *pli, Xio_ct_transform *transform)
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
