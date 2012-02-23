/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_load.h"
#include "dcmtk_series_set.h"
#include "file_util.h"
#include "rtds.h"

void
dcmtk_save_slice (Rtds *rtds, Volume *vol, const char *fn)
{
    char uid[100];
    DcmFileFormat fileformat;
    DcmDataset *dataset = fileformat.getDataset();
    dataset->putAndInsertString (DCM_SOPClassUID, 
        UID_SecondaryCaptureImageStorage);
    dataset->putAndInsertString (DCM_SOPInstanceUID, 
        dcmGenerateUniqueIdentifier (uid, SITE_INSTANCE_UID_ROOT));
    dataset->putAndInsertString (DCM_PatientName, "Doe^John");
    //dataset->putAndInsertUint8Array (DCM_PixelData, pixelData, pixelLength);
    OFCondition status = fileformat.saveFile (fn, EXS_LittleEndianExplicit);
    if (status.bad()) {
        print_and_exit ("Error: cannot write DICOM file (%s)\n", 
            status.text());
    }
}

void
dcmtk_save_image (Rtds *rtds, const char *dicom_dir)
{
    Volume *vol = rtds->m_img->gpuit_float();
    for (size_t k = 0; k < vol->dim[2]; k++) {
        Pstring p;
        p.format ("%s/image%03d.dcm", dicom_dir, (int) k);
        make_directory_recursive (p.c_str());
        dcmtk_save_slice (rtds, vol, p.c_str());
    }
}

void
dcmtk_save_rtds (Rtds *rtds, const char *dicom_dir)
{
    if (rtds->m_img) {
        dcmtk_save_image (rtds, dicom_dir);
    }
}
