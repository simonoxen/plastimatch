/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <string.h>
#include "dcmtk_config.h"
#include "dcmtk/dcmdata/dcvrda.h"
#include "dcmtk/dcmdata/dcvrtm.h"
#include "dcmtk/dcmdata/dctypes.h"
#include "dcmtk/dcmdata/dcuid.h"
#include "dcmtk/dcmdata/dcfilefo.h"
#include "dcmtk/dcmdata/dctag.h"
#include "dcmtk/dcmdata/dcdeftag.h"
#include "dicom_uid.h"

bool 
mondoshot_dicom_create_file (
		int height,
		int width,
		unsigned char* bytes,
		bool rgb,
		const char *patient_id,
		const char *patient_name,
		const char *filename)
{
    char uid[100];
    DcmFileFormat fileformat;
    DcmDataset *dataset = fileformat.getDataset();
    const char *uid_root = "1.2.826.0.1.3680043.8.274.1.1.200";
    
    dataset->putAndInsertString (DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
    dataset->putAndInsertString (DCM_SOPInstanceUID, plm_generate_dicom_uid (uid, uid_root));

    dataset->putAndInsertString (DCM_ConversionType, "WSD");
    dataset->putAndInsertString (DCM_ReferringPhysiciansName, "");
    dataset->putAndInsertString (DCM_PatientsName, patient_name);
    dataset->putAndInsertString (DCM_PatientID, patient_id);
    dataset->putAndInsertString (DCM_PatientsBirthDate, "");
    dataset->putAndInsertString (DCM_PatientsSex, "");

    /* These should be global for a session?? */
    dataset->putAndInsertString (DCM_StudyInstanceUID, plm_generate_dicom_uid (uid, uid_root));
    dataset->putAndInsertString (DCM_SeriesInstanceUID, plm_generate_dicom_uid (uid, uid_root));

    dataset->putAndInsertString (DCM_StudyID, "");
    dataset->putAndInsertString (DCM_SeriesNumber, "");
    dataset->putAndInsertString (DCM_InstanceNumber, "");
    dataset->putAndInsertString (DCM_PatientOrientation, "");

    if (rgb) {
	dataset->putAndInsertString (DCM_SamplesPerPixel, "3");
	dataset->putAndInsertString (DCM_PhotometricInterpretation, "RGB");
	dataset->putAndInsertString (DCM_PlanarConfiguration, "0");
    } else {
	dataset->putAndInsertString (DCM_SamplesPerPixel, "1");
	dataset->putAndInsertString (DCM_PhotometricInterpretation, "MONOCHROME2");
	dataset->putAndInsertString (DCM_PlanarConfiguration, "0");
    }
    dataset->putAndInsertUint16 (DCM_Rows, (Uint16) height);
    dataset->putAndInsertUint16 (DCM_Columns, (Uint16) width);
    dataset->putAndInsertString (DCM_BitsAllocated, "8");
    dataset->putAndInsertString (DCM_BitsStored, "8");
    dataset->putAndInsertString (DCM_HighBit, "7");
    dataset->putAndInsertString (DCM_PixelRepresentation, "0");

    /* At least we can set an instance creation Date/Time.  Should we set Study Date/Time too? */
    OFString s;
    DcmDate::getCurrentDate(s);
    dataset->putAndInsertOFStringArray(DCM_InstanceCreationDate, s);
    DcmTime::getCurrentTime(s);
    dataset->putAndInsertOFStringArray(DCM_InstanceCreationTime, s);

    if (rgb) {
	dataset->putAndInsertUint8Array (DCM_PixelData, bytes, height * width * 3);
    } else {
	dataset->putAndInsertUint8Array (DCM_PixelData, bytes, height * width);
    }

    OFCondition status = fileformat.saveFile (filename, EXS_LittleEndianExplicit);
    if (status.bad()) {
	// cerr << "Error: cannot write DICOM file (" << status.text() << ")" << endl;
	return false;
    }
    return true;
}
