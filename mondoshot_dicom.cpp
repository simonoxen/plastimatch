/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <string.h>
#include "dcmtk_config.h"
#include "dcmtk/dcmdata/dctypes.h"
#include "dcmtk/dcmdata/dcuid.h"
#include "dcmtk/dcmdata/dcfilefo.h"
#include "dcmtk/dcmdata/dctag.h"
#include "dcmtk/dcmdata/dcdeftag.h"
#include "dicom_uid.h"

bool 
mondoshot_dicom_send (int height,
		      int width,
		      unsigned char* bytes,
		      const char *patient_id,
		      const char *patient_name,
		      const char *dicom_local_ae,
		      const char *dicom_remote_ae,
		      const char *dicom_remote_host,
		      const char *dicom_remote_port)
{
    char uid[100];
    DcmFileFormat fileformat;
    DcmDataset *dataset = fileformat.getDataset();
    const char *uid_root = "1.2.826.0.1.3680043.8.274.1.1.200";
    

    dataset->putAndInsertString (DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
    dataset->putAndInsertString (DCM_SOPInstanceUID, plm_generate_dicom_uid (uid, uid_root));
    dataset->putAndInsertString (DCM_PatientsName, "Doe^John");

    dataset->putAndInsertUint8Array (DCM_PixelData, bytes, height * width * 3);
    OFCondition status = fileformat.saveFile ("test.dcm", EXS_LittleEndianExplicit);
    if (status.bad()) {
	// cerr << "Error: cannot write DICOM file (" << status.text() << ")" << endl;
	return false;
    }
    return true;
}
