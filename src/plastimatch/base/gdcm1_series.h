/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm1_series_h_
#define _gdcm1_series_h_

#include "plmbase_config.h"
#include "gdcmCommon.h"
#include "sys/plm_int.h"
#include <map>
#include <list>
#include <vector>

#include "pstring.h"

/* JAS 2012.05.06
 * Because this class uses a plm_long, we must include plm_int.h
 * Therefore, we must also include gdcmCommon.h here (even though
 * we don't need it) because, otherwise, the C99 integer fix imposed
 * by gdcmCommon.h will break and result in int8_t redefinition errors
 * under MSVC if other gdcm headers are subsequently included. */

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
