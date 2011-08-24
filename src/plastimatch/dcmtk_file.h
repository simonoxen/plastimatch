/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_file_h_
#define _dcmtk_file_h_

#include "plm_config.h"
#include "string_util.h"
#include "volume_header.h"

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
    Volume_header m_vh;
    
public:
    void debug () const;
    const char* get_cstr (const DcmTagKey& tag_key) const;
    void init ();
    void load (const char *fn);
};

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void
dcmtk_series_test (char *dicom_dir);

plastimatch1_EXPORT
bool
dcmtk_file_compare_z_position (const Dcmtk_file* f1, const Dcmtk_file* f2);

#if defined __cplusplus
}
#endif

#endif
