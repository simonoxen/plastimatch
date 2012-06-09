/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "plmbase.h"
#include "plmsys.h"

#include "dcmtk_file.h"
#include "dcmtk_save.h"
#include "dcmtk_series.h"
#include "dcmtk_uid.h"
#include "plm_math.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"

Plm_image*
Dcmtk_series::load_plm_image (void)
{
    /* Sort in Z direction */
    this->sort ();

    /* GCS FIX:
       (1) Direction cosines
       (2) Minimum 2 slices
       (3) Consistency of images w/in series
       (4) Rescale offset/slope
       (5) Different image types
       (6) Refine slice spacing based on entire chunk size
    */

    /* Check for minimum 2 slices */
    if (m_flist.size() < 2) {
	return 0;
    }
    
    /* Get first slice */
    std::list<Dcmtk_file*>::iterator it = m_flist.begin();
    Dcmtk_file *df = (*it);
    float z_init, z_prev, z_diff, z_last;
    int slice_no = 0;
    float best_chunk_z_start = z_init = z_prev = df->m_vh.m_origin[2];

    /* Get next slice */
    ++it; ++slice_no;
    df = (*it);
    z_diff = df->m_vh.m_origin[2] - z_prev;
    z_last = z_prev = df->m_vh.m_origin[2];

    /* We want to find the largest chunk with equal spacing.  This will 
       be used to resample in the case of irregular spacing. */
    int this_chunk_start = 0, best_chunk_start = 0;
    float this_chunk_diff = z_diff, best_chunk_diff = z_diff;
    int this_chunk_len = 2, best_chunk_len = 2;

    /* Loop through remaining slices */
    while (++it != m_flist.end())
    {
	++slice_no;
	df = (*it);
	z_diff = df->m_vh.m_origin[2] - z_prev;
	z_last = z_prev = df->m_vh.m_origin[2];

	if (fabs (this_chunk_diff - z_diff) > 0.11) {
	    /* Start a new chunk if difference in thickness is 
	       more than 0.1 millimeter */
	    this_chunk_start = slice_no - 1;
	    this_chunk_len = 2;
	    this_chunk_diff = z_diff;
	} else {
	    /* Same thickness, increase size of this chunk */
	    this_chunk_diff = ((this_chunk_len * this_chunk_diff) + z_diff)
		/ (this_chunk_len + 1);
	    this_chunk_len++;

	    /* Check if this chunk is now the best chunk */
	    if (this_chunk_len > best_chunk_len) {
		best_chunk_start = this_chunk_start;
		best_chunk_len = this_chunk_len;
		best_chunk_diff = this_chunk_diff;
		best_chunk_z_start = z_prev 
		    - (best_chunk_len-1) * best_chunk_diff;
	    }
	}
    }

    /* Report information about best chunk */
    printf ("Best chuck:\n  Slices %d to %d from (0 to %d)\n"
	"  Z_loc = %f %f\n" 
	"  Slice spacing = %f\n", 
	best_chunk_start, best_chunk_start + best_chunk_len - 1, slice_no, 
	best_chunk_z_start, 
	best_chunk_z_start + (best_chunk_len - 1) * best_chunk_diff, 
	best_chunk_diff);

    /* Some debugging info */
    printf ("Slices: ");
    for (it = m_flist.begin(); it != m_flist.end(); ++it) {
	Dcmtk_file *df = (*it);
	printf ("%f ", df->m_vh.m_origin[2]);
    }
    printf ("\n");

    /* Compute resampled volume header */
    Volume_header vh;
    int slices_before = 
	ROUND_INT ((best_chunk_z_start - z_init) / best_chunk_diff);
    int slices_after = 
	ROUND_INT ((z_last - best_chunk_z_start 
		- (best_chunk_len - 1) * best_chunk_diff) / best_chunk_diff);
    df = (*m_flist.begin());
    vh.clone (&df->m_vh);
    vh.m_dim[2] = slices_before + best_chunk_len + slices_after;
    vh.m_origin[2] = best_chunk_z_start - slices_before * best_chunk_diff;
    vh.m_spacing[2] = best_chunk_diff;

    /* More debugging info */
    vh.print ();

    /* Still more debugging info */
    printf ("Resamples slices: ");
    for (plm_long i = 0; i < vh.m_dim[2]; i++) {
	printf ("%f ", vh.m_origin[2] + i * vh.m_spacing[2]);
    }
    printf ("\n");

    /* Divine image type */
    df = (*m_flist.begin());
    uint16_t samp_per_pix, bits_alloc, bits_stored, high_bit, pixel_rep;
    const char* phot_interp;
    bool rc = df->get_uint16 (DCM_SamplesPerPixel, &samp_per_pix);
    if (!rc) {
	return 0;
    }
    phot_interp = df->get_cstr (DCM_PhotometricInterpretation);
    if (!phot_interp) {
	return 0;
    }
    rc = df->get_uint16 (DCM_BitsAllocated, &bits_alloc);
    if (!rc) {
	return 0;
    }
    rc = df->get_uint16 (DCM_BitsStored, &bits_stored);
    if (!rc) {
	return 0;
    }
    rc = df->get_uint16 (DCM_HighBit, &high_bit);
    if (!rc) {
	return 0;
    }
    rc = df->get_uint16 (DCM_PixelRepresentation, &pixel_rep);
    if (!rc) {
	return 0;
    }
    printf ("Samp_per_pix: %d\n", (int) samp_per_pix);
    printf ("Phot_interp: %s\n", phot_interp);
    printf ("Bits_alloc: %d\n", (int) bits_alloc);
    printf ("Bits_stored: %d\n", (int) bits_stored);
    printf ("High_bit: %d\n", (int) high_bit);
    printf ("Pixel_rep: %d\n", (int) pixel_rep);

    /* Some kinds of images we don't know how to deal with.  
       Don't load these. */
    if (samp_per_pix != 1) {
	return 0;
    }
    if (strcmp (phot_interp, "MONOCHROME2")) {
	return 0;
    }
    if (bits_alloc != 16) {
	return 0;
    }
    if (bits_stored != 16) {
	return 0;
    }
    if (high_bit != 15) {
	return 0;
    }
    if (pixel_rep != 1) {
	return 0;
    }

    printf ("Image looks ok.  Try to load.\n");

    Plm_image *pli = new Plm_image;
    pli->m_type = PLM_IMG_TYPE_GPUIT_SHORT;
    pli->m_original_type = PLM_IMG_TYPE_GPUIT_SHORT;
    pli->m_gpuit = new Volume (vh, PT_SHORT, 1);
    Volume* vol = (Volume*) pli->m_gpuit;
    uint16_t* img = (uint16_t*) vol->img;

    for (plm_long i = 0; i < vh.m_dim[2]; i++) {
	/* Find the best slice, using nearest neighbor interpolation */
	std::list<Dcmtk_file*>::iterator best_slice_it = m_flist.begin();
	float best_z_dist = FLT_MAX;
	float z_pos = vh.m_origin[2] + i * vh.m_spacing[2];
	for (it = m_flist.begin(); it != m_flist.end(); ++it) {
	    float this_z_dist = fabs ((*it)->m_vh.m_origin[2] - z_pos);
	    if (this_z_dist < best_z_dist) {
		best_z_dist = this_z_dist;
		best_slice_it = it;
	    }
	}

	/* Load the slice image data into volume */
	const uint16_t* pixel_data;
	Dcmtk_file *df = (*best_slice_it);
	unsigned long length;

	printf ("Loading slice z=%f at location z=%f\n",
	    (*best_slice_it)->m_vh.m_origin[2], z_pos);

	rc = df->get_uint16_array (DCM_PixelData, &pixel_data, &length);
	if (!rc) {
	    print_and_exit ("Oops.  Error reading pixel data.  Punting.\n");
	}
	if (((long) length) != vh.m_dim[0] * vh.m_dim[1]) {
	    print_and_exit ("Oops.  Dicom image had wrong length "
		"(%d vs. %d x %d).\n", length, vh.m_dim[0],
		vh.m_dim[1]);
	}
	memcpy (img, pixel_data, length * sizeof(uint16_t));
	img += length;
    }
    return pli;
}

static void
dcmtk_save_slice (const Dcmtk_study_writer *dsw, Dcmtk_slice_data *dsd)
{
    Pstring tmp;
    DcmFileFormat fileformat;
    DcmDataset *dataset = fileformat.getDataset();

    dataset->putAndInsertString (DCM_ImageType, 
        "DERIVED\\SECONDARY\\REFORMATTED");
    dataset->putAndInsertOFStringArray(DCM_InstanceCreationDate, 
        dsw->date_string);
    dataset->putAndInsertOFStringArray(DCM_InstanceCreationTime, 
        dsw->time_string);
    dataset->putAndInsertString (DCM_SOPClassUID, UID_CTImageStorage);
    dataset->putAndInsertString (DCM_SOPInstanceUID, dsd->slice_uid);
    dataset->putAndInsertOFStringArray (DCM_StudyDate, dsw->date_string);
    dataset->putAndInsertOFStringArray (DCM_StudyTime, dsw->time_string);
    dataset->putAndInsertString (DCM_AccessionNumber, "");
    dataset->putAndInsertString (DCM_Modality, "CT");
    dataset->putAndInsertString (DCM_Manufacturer, "Plastimatch");
    dataset->putAndInsertString (DCM_ReferringPhysicianName, "");
    dataset->putAndInsertString (DCM_PatientName, "");
    dataset->putAndInsertString (DCM_PatientID, "");
    dataset->putAndInsertString (DCM_PatientBirthDate, "");
    dataset->putAndInsertString (DCM_PatientSex, "");
    dataset->putAndInsertString (DCM_SliceThickness, dsd->sthk.c_str());
    dataset->putAndInsertString (DCM_SoftwareVersions,
        PLASTIMATCH_VERSION_STRING);
    /* GCS FIX: PatientPosition */
    dataset->putAndInsertString (DCM_PatientPosition, "HFS");
    dataset->putAndInsertString (DCM_StudyInstanceUID, dsw->study_uid);
    dataset->putAndInsertString (DCM_SeriesInstanceUID, dsw->ct_series_uid);
    dataset->putAndInsertString (DCM_StudyID, "10001");
    dataset->putAndInsertString (DCM_SeriesNumber, "303");
    dataset->putAndInsertString (DCM_InstanceNumber, "0");
    /* GCS FIX: PatientOrientation */
    dataset->putAndInsertString (DCM_PatientOrientation, "L/P");
    dataset->putAndInsertString (DCM_ImagePositionPatient, dsd->ipp);
    dataset->putAndInsertString (DCM_ImageOrientationPatient, dsd->iop);
    dataset->putAndInsertString (DCM_FrameOfReferenceUID, dsw->for_uid);
    dataset->putAndInsertString (DCM_SliceLocation, dsd->sloc.c_str());
    dataset->putAndInsertString (DCM_SamplesPerPixel, "1");
    dataset->putAndInsertString (DCM_PhotometricInterpretation, "MONOCHROME2");
    dataset->putAndInsertUint16 (DCM_Rows, (Uint16) dsd->vol->dim[1]);
    dataset->putAndInsertUint16 (DCM_Columns, (Uint16) dsd->vol->dim[0]);
    tmp.format ("%f\\%f", dsd->vol->spacing[0], dsd->vol->spacing[1]);
    dataset->putAndInsertString (DCM_PixelSpacing, tmp.c_str());
    dataset->putAndInsertString (DCM_BitsAllocated, "16");
    dataset->putAndInsertString (DCM_BitsStored, "16");
    dataset->putAndInsertString (DCM_HighBit, "15");
    dataset->putAndInsertString (DCM_PixelRepresentation, "1");
    dataset->putAndInsertString (DCM_RescaleIntercept, "0");
    dataset->putAndInsertString (DCM_RescaleSlope, "1");
    dataset->putAndInsertString (DCM_RescaleType, "US");

    /* Convert to 16-bit signed int */
    for (size_t i = 0; i < dsd->slice_size; i++) {
        float f = dsd->slice_float[i];
        dsd->slice_int16[i] = (int16_t) f;
    }

    dataset->putAndInsertUint16Array (DCM_PixelData, 
        (Uint16*) dsd->slice_int16, dsd->slice_size);
    OFCondition status = fileformat.saveFile (dsd->fn.c_str(), 
        EXS_LittleEndianExplicit);
    if (status.bad()) {
        print_and_exit ("Error: cannot write DICOM file (%s)\n", 
            status.text());
    }
}

void
Dcmtk_save::save_image (
    Dcmtk_study_writer *dsw, 
    const char *dicom_dir)
{
    Dcmtk_slice_data dsd;
//    dsd.rtds = rtds;
    dsd.vol = this->img->gpuit_float();

    dsd.slice_size = dsd.vol->dim[0] * dsd.vol->dim[1];
    dsd.slice_int16 = new int16_t[dsd.slice_size];
    float *dc = dsd.vol->direction_cosines.m_direction_cosines;
    dsd.iop.format ("%f\\%f\\%f\\%f\\%f\\%f",
        dc[0], dc[1], dc[2], dc[3], dc[4], dc[5]);

    for (plm_long k = 0; k < dsd.vol->dim[2]; k++) {
        dsd.fn.format ("%s/image%03d.dcm", dicom_dir, (int) k);
        make_directory_recursive (dsd.fn);
        /* GCS FIX: direction cosines */
        dsd.sthk.format ("%f", dsd.vol->spacing[2]);
        dsd.sloc.format ("%f", dsd.vol->offset[2] + k * dsd.vol->spacing[2]);
        dsd.ipp.format ("%f\\%f\\%f", dsd.vol->offset[0], dsd.vol->offset[1], 
            dsd.vol->offset[2] + k * dsd.vol->spacing[2]);
        dcmtk_uid (dsd.slice_uid, PLM_UID_PREFIX);

        dsd.slice_float = &((float*)dsd.vol->img)[k*dsd.slice_size];
        dcmtk_save_slice (dsw, &dsd);
        dsw->slice_data.push_back (dsd);
    }
    delete[] dsd.slice_int16;
}
