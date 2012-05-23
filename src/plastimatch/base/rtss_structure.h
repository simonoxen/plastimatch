/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtss_structure_h_
#define _rtss_structure_h_

#include "plmbase_config.h"
#include "pstring.h"
#include "plmbase_config.h"

#define CXT_BUFLEN 2048

class Plm_image;
class Plm_image_header;

class Rtss_polyline {
public:
    int slice_no;           /* Can be "-1" */
    Pstring ct_slice_uid;
    int num_vertices;
    float* x;
    float* y;
    float* z;
public:
    PLMBASE_API Rtss_polyline ();
    PLMBASE_API ~Rtss_polyline ();
};

class Rtss_structure {
public:
    Pstring name;
    Pstring color;
    int id;                    /* Used for import/export (must be >= 1) */
    int bit;                   /* Used for ss-img (-1 for no bit) */
    size_t num_contours;
    Rtss_polyline** pslist;
public:
    PLMBASE_API Rtss_structure ();
    PLMBASE_API ~Rtss_structure ();

    PLMBASE_API void clear ();
    PLMBASE_API Rtss_polyline* add_polyline ();
    PLMBASE_API void set_color (const char* color_string);
    PLMBASE_API void get_dcm_color_string (Pstring *dcm_color) const;
    PLMBASE_API void structure_rgb (int *r, int *g, int *b) const;

    static void adjust_name (Pstring *name_out, const Pstring *name_in);
};


#endif
