/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef __mondoshot_dicom_h__
#define __mondoshot_dicom_h__

bool 
mondoshot_dicom_create_file (
		int height,
		int width,
		unsigned char* bytes,
		bool rgb,
		const char *patient_id,
		const char *patient_name,
		const char *dicom_local_ae,
		const char *dicom_remote_ae,
		const char *dicom_remote_host,
		const char *dicom_remote_port,
		const char *filename);


#endif
