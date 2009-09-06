/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _file_type_h_
#define _file_type_h_

enum File_type {
    FILE_TYPE_NO_FILE,
    FILE_TYPE_UNKNOWN,
    FILE_TYPE_IMG,
    FILE_TYPE_DIJ,
    FILE_TYPE_POINTSET,
    FILE_TYPE_CXT,
    FILE_TYPE_DICOM_DIR,
    FILE_TYPE_XIO_DIR,
    FILE_TYPE_RTOG_DIR,
};

File_type
deduce_file_type (char* path);
char*
file_type_string (File_type);

#endif
