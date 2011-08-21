/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_file_h_
#define _dcmtk_file_h_

#include "plm_config.h"

class DcmFile;
class DcmDataset;
class DcmTagKey;

class Dcmtk_file
{
public:
    Dcmtk_file ();
    Dcmtk_file (const char *fn);
    ~Dcmtk_file ();

public:
    std::string m_fn;
    DcmFileFormat *m_dfile;
    
public:
    void init ();
    void debug () const;
    void load (const char *fn);
    const char* get_cstr (const DcmTagKey& tag_key);

#if defined (commentout)
    void load (const char *dicom_dir);
    void digest_files (void);
    void get_slice_info (int *slice_no, Pstring *ct_slice_uid, float z);
    gdcm::File *get_ct_slice (void);
    void get_slice_uids (std::vector<Pstring> *slice_uids);
    std::string get_patient_position ();
    const std::string& get_rtdose_filename ();
    const std::string& get_rtstruct_filename ();
    void get_img_metadata (Img_metadata *img_metadata);

    gdcm::SerieHelper2 *m_gsh2;

    int m_have_ct;
    gdcm::FileList *m_ct_file_list;
    gdcm::FileList *m_rtdose_file_list;
    gdcm::FileList *m_rtstruct_file_list;
    int m_dim[3];
    double m_origin[3];
    double m_spacing[3];
#endif
};

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void
dcmtk_series_test (char *dicom_dir);

#if defined __cplusplus
}
#endif

#endif
