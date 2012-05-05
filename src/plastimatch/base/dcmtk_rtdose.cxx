/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "plmsys.h"

#include "dcmtk_file.h"
#include "dcmtk_series.h"
#include "plm_math.h"
#include "rtds.h"

/* This is the tolerance on irregularity of the grid spacing (in mm) */
#define GFOV_SPACING_TOL (1e-1)

template <class T> 
void
dcmtk_dose_copy (float *img_out, T *img_in, int nvox, float scale)
{
    for (int i = 0; i < nvox; i++) {
	img_out[i] = img_in[i] * scale;
    }
}

void
Dcmtk_series::rtdose_load (
    Rtds *rtds                       /* Output: this gets updated */
)
{
    int rc;
    const char *val;
    uint16_t val_u16;
    plm_long dim[3];
    float ipp[3];
    float spacing[3];
    float *gfov;    /* gfov = GridFrameOffsetVector */
    plm_long gfov_len;
    const char *gfov_str;

    /* Modality -- better be RTDOSE */
    std::string modality = this->get_modality();
    if (modality == "RTDOSE") {
        printf ("Trying to load rt dose.\n");
    } else {
        print_and_exit ("Oops.\n");
    }

    /* FIX: load metadata such as patient name, etc. */

    /* ImagePositionPatient */
    val = this->get_cstr (DCM_ImagePositionPatient);
    if (!val) {
        print_and_exit ("Couldn't find DCM_ImagePositionPatient in rtdose\n");
    }
    rc = sscanf (val, "%f\\%f\\%f", &ipp[0], &ipp[1], &ipp[2]);
    if (rc != 3) {
	print_and_exit ("Error parsing RTDOSE ipp.\n");
    }

    /* Rows */
    if (!this->get_uint16 (DCM_Rows, &val_u16)) {
        print_and_exit ("Couldn't find DCM_Rows in rtdose\n");
    }
    dim[1] = val_u16;

    /* Columns */
    if (!this->get_uint16 (DCM_Columns, &val_u16)) {
        print_and_exit ("Couldn't find DCM_Columns in rtdose\n");
    }
    dim[0] = val_u16;

    /* PixelSpacing */
    val = this->get_cstr (DCM_PixelSpacing);
    if (!val) {
        print_and_exit ("Couldn't find DCM_PixelSpacing in rtdose\n");
    }
    rc = sscanf (val, "%g\\%g", &spacing[1], &spacing[0]);
    if (rc != 2) {
	print_and_exit ("Error parsing RTDOSE pixel spacing.\n");
    }

    /* GridFrameOffsetVector */
    val = this->get_cstr (DCM_GridFrameOffsetVector);
    if (!val) {
        print_and_exit ("Couldn't find DCM_GridFrameOffsetVector in rtdose\n");
    }
    gfov = 0;
    gfov_len = 0;
    gfov_str = val;
    while (1) {
	int len;
	gfov = (float*) realloc (gfov, (gfov_len + 1) * sizeof(float));
	rc = sscanf (gfov_str, "%g%n", &gfov[gfov_len], &len);
	if (rc != 1) {
	    break;
	}
	gfov_len ++;
	gfov_str += len;
	if (gfov_str[0] == '\\') {
	    gfov_str ++;
	}
    }
    dim[2] = gfov_len;
    if (gfov_len == 0) {
	print_and_exit ("Error parsing RTDOSE gfov.\n");
    }

    /* --- Analyze GridFrameOffsetVector --- */

    /* (1) Make sure first element is 0. */
    if (gfov[0] != 0.) {
	if (gfov[0] == ipp[2]) {
	    /* In this case, gfov values are absolute rather than relative 
	       positions, but we process the same way. */
	} else {
	    /* This is wrong.  But Nucletron does it. */
	    logfile_printf (
		"Warning: RTDOSE gfov[0] is neither 0 nor ipp[2].\n"
		"This violates the DICOM standard.  Proceeding anyway...\n");
	    /* Nucletron seems to work by ignoring absolute offset (???) */
	}
    }

    /* (2) Handle case where gfov_len == 1 (only one slice). */
    if (gfov_len == 1) {
	spacing[2] = spacing[0];
    }

    /* (3) Check to make sure spacing is regular. */
    for (plm_long i = 1; i < gfov_len; i++) {
	if (i == 1) {
	    spacing[2] = gfov[1] - gfov[0];
	} else {
	    float sp = gfov[i] - gfov[i-1];
	    if (fabs(sp - spacing[2]) > GFOV_SPACING_TOL) {
		print_and_exit ("Error RTDOSE grid has irregular spacing:"
		    "%f vs %f.\n", sp, spacing[2]);
	    }
	}
    }

    /* DoseGridScaling -- if element doesn't exist, scaling is 1.0 */
    float dose_scaling = 1.0;
    val = this->get_cstr (DCM_DoseGridScaling);
    if (val) {
        /* No need to check for success, let scaling be 1.0 if failure */
        sscanf (val, "%f", &dose_scaling);
    }

    printf ("RTDOSE: dim = %d %d %d\n        %f %f %f\n        %f %f %f\n",
        (int) dim[0], (int) dim[1], (int) dim[2],
        ipp[0], ipp[1], ipp[2], 
        spacing[0], spacing[1], spacing[2]);

    uint16_t bits_alloc, bits_stored, high_bit, pixel_rep;
    rc = this->get_uint16 (DCM_BitsAllocated, &bits_alloc);
    if (!rc) {
        print_and_exit ("Couldn't find DCM_BitsAllocated in rtdose\n");
    }
    rc = this->get_uint16 (DCM_BitsStored, &bits_stored);
    if (!rc) {
        print_and_exit ("Couldn't find DCM_BitsStored in rtdose\n");
    }
    rc = this->get_uint16 (DCM_HighBit, &high_bit);
    if (!rc) {
        print_and_exit ("Couldn't find DCM_HighBit in rtdose\n");
    }
    rc = this->get_uint16 (DCM_PixelRepresentation, &pixel_rep);
    if (!rc) {
        print_and_exit ("Couldn't find DCM_PixelRepresentation in rtdose\n");
    }

    printf ("Bits_alloc: %d\n", (int) bits_alloc);
    printf ("Bits_stored: %d\n", (int) bits_stored);
    printf ("High_bit: %d\n", (int) high_bit);
    printf ("Pixel_rep: %d\n", (int) pixel_rep);

    /* Create output dose image */
    Plm_image *pli = new Plm_image;
    rtds->set_dose (pli);

    /* Create Volume */
    Volume *vol = new Volume (dim, ipp, spacing, 0, PT_FLOAT, 1);
    float *img = (float*) vol->img;

    /* Bind volume to plm_image */
    pli->set_gpuit (vol);

    /* PixelData */
    unsigned long length = 0;
    if (pixel_rep == 0) {
        const uint16_t* pixel_data;
        rc = this->m_flist.front()->get_uint16_array (
            DCM_PixelData, &pixel_data, &length);
        printf ("rc = %d, length = %lu, npix = %ld\n", 
            rc, length, (long) vol->npix);
        if (bits_stored == 16) {
            dcmtk_dose_copy (img, (const uint16_t*) pixel_data, 
                vol->npix, dose_scaling);
        } else if (bits_stored == 32) {
            dcmtk_dose_copy (img, (const uint32_t*) pixel_data, 
                vol->npix, dose_scaling);
        } else {
            delete pli;
            print_and_exit ("Unknown pixel representation (%d %d)\n",
                bits_stored, pixel_rep);
        }
    } else {
        const int16_t* pixel_data;
        rc = this->m_flist.front()->get_int16_array (
            DCM_PixelData, &pixel_data, &length);
        if (bits_stored == 16) {
            dcmtk_dose_copy (img, (const int16_t*) pixel_data, 
                vol->npix, dose_scaling);
        } else if (bits_stored == 32) {
            dcmtk_dose_copy (img, (const int32_t*) pixel_data, 
                vol->npix, dose_scaling);
        } else {
            delete pli;
            print_and_exit ("Unknown pixel representation (%d %d)\n",
                bits_stored, pixel_rep);
        }
    }

#if defined (commentout)
#endif
}
