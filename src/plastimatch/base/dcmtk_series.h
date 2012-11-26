/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_series_h_
#define _dcmtk_series_h_

#include "plmbase_config.h"
#include <list>

#include "plm_int.h"
#include "pstring.h"

class DcmTagKey;

class Dcmtk_file;
class Dcmtk_series_private;
class Dicom_rt_study;
class Plm_image;

class Dcmtk_series 
{
public:
    Dcmtk_series ();
    ~Dcmtk_series ();

public:
    Dcmtk_series_private *d_ptr;

public:
    const char* get_cstr (const DcmTagKey& tag_key) const;
    bool get_int16_array (const DcmTagKey& tag_key, 
        const int16_t** val, unsigned long* count) const;
    bool get_sequence (const DcmTagKey& tag_key, 
        DcmSequenceOfItems *seq) const;
    std::string get_string (const DcmTagKey& tag_key) const;
    bool get_uint16 (const DcmTagKey& tag_key, uint16_t* val) const;
    bool get_uint16_array (const DcmTagKey& tag_key, 
        const uint16_t** val, unsigned long* count) const;

    std::string get_modality (void) const;
    std::string get_referenced_uid (void) const;

    void insert (Dcmtk_file* df);
    void sort (void);

    void set_rt_study (Dicom_rt_study *drs);
    Plm_image *load_plm_image ();

    void debug (void) const;
};

#endif
