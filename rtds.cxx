/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cxt_apply_dicom.h"
#include "cxt_extract.h"
#include "gdcm_rtss.h"
#include "rtds.h"
#include "xio_ct.h"
#include "xio_dir.h"
#include "xio_io.h"
#include "xio_structures.h"

void
Rtds::load_dicom_dir (char *dicom_dir)
{
    this->m_img = plm_image_load_native (dicom_dir);
    
}

void
Rtds::load_xio (char *xio_dir, char *dicom_dir)
{
    Xio_dir *xd;
    Xio_patient_dir *xpd;
    Xio_studyset_dir *xsd;

    xd = xio_dir_create (xio_dir);

    if (xd->num_patient_dir <= 0) {
	print_and_exit ("Error, xio num_patient_dir = %d\n", 
	    xd->num_patient_dir);
    }
    xpd = &xd->patient_dir[0];
    if (xd->num_patient_dir > 1) {
	printf ("Warning: multiple patients found in xio directory.\n"
	    "Defaulting to first directory: %s\n", xpd->path);
    }
    if (xpd->num_studyset_dir <= 0) {
	print_and_exit ("Error, xio patient has no studyset.");
    }
    xsd = &xpd->studyset_dir[0];
    if (xpd->num_studyset_dir > 1) {
	printf ("Warning: multiple studyset found in xio patient directory.\n"
	    "Defaulting to first directory: %s\n", xsd->path);
    }

    /* Load the XiO CT images */
    this->m_img = new Plm_image;
    xio_ct_load (this->m_img, xsd->path);

    /* Load the XiO structure set */
    this->m_cxt = cxt_create ();
    printf ("calling xio_structures_load\n");
    xio_structures_load (this->m_cxt, xsd->path);

    /* Apply XiO CT geometry to structures */
    printf ("calling cxt_set_geometry_from_plm_image\n");
    cxt_set_geometry_from_plm_image (this->m_cxt, this->m_img);

    /* If a directory with original DICOM CT is provided,
       the UIDs of the matching slices will be added to structures
       and the coordinates will be transformed from XiO to DICOM LPS. */

    if (dicom_dir[0]) {
	/* Transform CT from XiO coordiantes to DICOM LPS */
	xio_ct_apply_dicom_dir (this->m_img, dicom_dir);

	/* Transform structures from XiO coordinates to DICOM LPS */
	xio_structures_apply_dicom_dir (this->m_cxt, dicom_dir);

	/* Associate structures with DICOM */
	cxt_apply_dicom_dir (this->m_cxt, dicom_dir);
    }
}

void
Rtds::load_ss_img (char *ss_img, char *ss_list)
{
    /* Load ss_img */
    if (this->m_ss_img) {
	delete this->m_ss_img;
    }
    if (ss_img) {
	this->m_ss_img = plm_image_load_native (ss_img);
    }

    /* Load ss_list */
    if (this->m_ss_list) {
	cxt_destroy (this->m_ss_list);
    }
    if (ss_list) {
	this->m_ss_list = cxt_load_ss_list (0, ss_list);
    }
}

void
Rtds::save_dicom (char *dicom_dir)
{
    if (this->m_img) {
	this->m_img->save_short_dicom (dicom_dir);
    }
}

void
Rtds::convert_ss_img_to_cxt (void)
{
    int num_structs = -1;

    /* Allocate memory for cxt */
    if (this->m_cxt) {
	cxt_destroy (this->m_cxt);
    }
    this->m_cxt = cxt_create ();

    /* Copy geometry from ss_img to cxt */
    cxt_set_geometry_from_plm_image (this->m_cxt, this->m_ss_img);

    /* Extract polylines */
	num_structs = this->m_ss_list->num_structures;
    /* Image type must be uint32_t for cxt_extract */
    this->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);

    /* Do extraction */
    printf ("Running marching squares\n");
    if (this->m_ss_list) {
	cxt_clone_empty (this->m_cxt, this->m_ss_list);
	cxt_extract (this->m_cxt, this->m_ss_img->m_itk_uint32, -1, true);
    } else {
	cxt_extract (this->m_cxt, this->m_ss_img->m_itk_uint32, -1, false);
    }
}
