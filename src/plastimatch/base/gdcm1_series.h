/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm1_series_h_
#define _gdcm1_series_h_

#include "plmbase_config.h"
#include <map>
#include <list>
#include <vector>

#include "plmsys.h"

#include "pstring.h"

class Metadata;

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
    void get_slice_info (int *slice_no, Pstring *ct_slice_uid, float z);
    gdcm::File *get_ct_slice (void);
    void get_slice_uids (std::vector<Pstring> *slice_uids);
    std::string get_patient_position ();
    const std::string& get_rtdose_filename ();
    const std::string& get_rtstruct_filename ();
    void get_metadata (Metadata *meta);

    gdcm::SerieHelper2 *m_gsh2;

    int m_have_ct;
    gdcm::FileList *m_ct_file_list;
    gdcm::FileList *m_rtdose_file_list;
    gdcm::FileList *m_rtstruct_file_list;
    plm_long m_dim[3];
    double m_origin[3];
    double m_spacing[3];
};
#endif
