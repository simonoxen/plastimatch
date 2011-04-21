/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm_series_h_
#define _gdcm_series_h_

#include "plm_config.h"
#include <map>
#include <list>
#include <vector>
#include "bstrwrap.h"
#include "img_metadata.h"

/* Forward declarations */
namespace gdcm {
    class File;
    typedef std::vector<File* > FileList;
    class SerieHelper2;
};

class Gdcm_series 
{
public:
    Gdcm_series ();
    ~Gdcm_series ();

    void load (const char *dicom_dir);
    void digest_files (void);
    void get_slice_info (int *slice_no, CBString *ct_slice_uid, float z);
    gdcm::File *get_ct_slice (void);
    void get_slice_uids (std::vector<CBString> *slice_uids);
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
};

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void
gdcm_series_test (char *dicom_dir);

#if defined __cplusplus
}
#endif

#endif
