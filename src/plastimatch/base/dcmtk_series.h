/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_series_h_
#define _dcmtk_series_h_

#include "plmbase_config.h"
#include <list>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "pstring.h"

class DcmTagKey;

class Dcmtk_file;
class Plm_image;
//class Rtds;
class Volume;

typedef
struct dcmtk_slice_data
{
    Pstring fn;
//    Rtds *rtds;
    Volume *vol;

    size_t slice_size;
    float *slice_float;
    int16_t *slice_int16;

    char slice_uid[100];
    Pstring ipp;
    Pstring iop;
    Pstring sloc;
    Pstring sthk;
} Dcmtk_slice_data;

class Dcmtk_study_writer {
public:
    OFString date_string;
    OFString time_string;
    char study_uid[100];
    char for_uid[100];
    char ct_series_uid[100];
    char rtss_instance_uid[100];
    char rtss_series_uid[100];
    std::vector<Dcmtk_slice_data> slice_data;
};

class Dcmtk_series 
{
public:
    Dcmtk_series ();
    ~Dcmtk_series ();

public:
    
    std::list<Dcmtk_file*> m_flist;

public:
    void debug (void) const;
    const char* get_cstr (const DcmTagKey& tag_key) const;
    std::string get_string (const DcmTagKey& tag_key) const;
    bool get_uint16 (const DcmTagKey& tag_key, uint16_t* val) const;
    std::string get_modality (void) const;
    std::string get_referenced_uid (void) const;
    void insert (Dcmtk_file* df);
    Plm_image* load_plm_image ();
    void sort (void);
};

#endif
