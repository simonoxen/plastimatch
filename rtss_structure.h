/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtss_structure_h_
#define _rtss_structure_h_

#include "plm_config.h"
#include "bstrwrap.h"

#include "demographics.h"
#include "plm_image.h"
#include "plm_image_header.h"

#define CXT_BUFLEN 2048

class Rtss_polyline {
public:
    int slice_no;
    CBString ct_slice_uid;
    int num_vertices;
    float* x;
    float* y;
    float* z;
public:
    plastimatch1_EXPORT
    Rtss_polyline ();
    plastimatch1_EXPORT
    ~Rtss_polyline ();
};

class Rtss_structure {
public:
    CBString name;
    CBString color;
    int id;                    /* Used for import/export (must be >= 1) */
    int bit;                   /* Used for ss-img (-1 for no bit) */
    int num_contours;
    Rtss_polyline** pslist;
public:
    plastimatch1_EXPORT
    Rtss_structure ();
    plastimatch1_EXPORT
    ~Rtss_structure ();

    void clear ();
    Rtss_polyline* add_polyline ();
    void set_color (const char* color_string);
    void get_dcm_color_string (CBString *dcm_color) const;
    void structure_rgb (int *r, int *g, int *b) const;

    static void adjust_name (CBString *name_out, const CBString *name_in);
};

#if defined __cplusplus
extern "C" {
#endif

#if defined (commentout)
plastimatch1_EXPORT
Cxt_structure_list*
cxt_create (void);
plastimatch1_EXPORT
void
cxt_init (Cxt_structure_list* structures);
plastimatch1_EXPORT
Cxt_structure*
cxt_add_structure (
    Cxt_structure_list *cxt, 
    const CBString& structure_name, 
    const CBString& color, 
    int structure_id);
plastimatch1_EXPORT
Cxt_structure*
cxt_find_structure_by_id (Cxt_structure_list* structures, int structure_id);
plastimatch1_EXPORT
void
cxt_debug (Cxt_structure_list* structures);
plastimatch1_EXPORT
void
cxt_adjust_structure_names (Cxt_structure_list* structures);
plastimatch1_EXPORT
void
cxt_free (Cxt_structure_list* structures);
plastimatch1_EXPORT
void
cxt_destroy (Cxt_structure_list* structures);
plastimatch1_EXPORT
void
cxt_prune_empty (Cxt_structure_list* structures);
plastimatch1_EXPORT
Cxt_structure_list*
cxt_clone_empty (
    Cxt_structure_list* cxt_out, 
    Cxt_structure_list* cxt_in
);
plastimatch1_EXPORT
void
cxt_apply_geometry (Cxt_structure_list* structures);
plastimatch1_EXPORT
void
cxt_set_geometry_from_plm_image_header (
    Cxt_structure_list* cxt, 
    Plm_image_header *pih
);
plastimatch1_EXPORT
void
cxt_set_geometry_from_plm_image (
    Cxt_structure_list* structures,
    Plm_image *pli
);
#endif

#if defined (commentout)
plastimatch1_EXPORT
Cxt_polyline*
cxt_add_polyline (Cxt_structure* structure);
plastimatch1_EXPORT
void
cxt_structure_rgb (const Cxt_structure *structure, int *r, int *g, int *b);
plastimatch1_EXPORT
void
cxt_adjust_name (CBString *name_out, const CBString *name_in);
#endif

#if defined __cplusplus
}
#endif

#endif
