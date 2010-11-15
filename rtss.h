/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtss_h_
#define _rtss_h_

#include "plm_config.h"
#include <list>
#include "bstrwrap.h"

#include "demographics.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "rtss_structure.h"

#define CXT_BUFLEN 2048

class Rtss {
public:
    Demographics *m_demographics;
    CBString ct_study_uid;
    CBString ct_series_uid;
    CBString ct_fref_uid;
    CBString study_id;
    std::list<CBString> ct_slice_uids;
    int have_geometry;
    int dim[3];
    float spacing[3];
    float offset[3];
    int num_structures;
    Rtss_structure **slist;
public:
    plastimatch1_EXPORT
    Rtss ();
    plastimatch1_EXPORT
    ~Rtss ();
    void init (void);
    void clear (void);
    Rtss_structure* add_structure (
	const CBString& structure_name, 
	const CBString& color, 
	int structure_id);
    Rtss_structure* find_structure_by_id (int structure_id);
    void debug (void);
    void adjust_structure_names (void);
    void prune_empty (void);
    static Rtss* clone_empty (Rtss* cxt_out, Rtss* cxt_in);
    plastimatch1_EXPORT
    void free_all_polylines (void);
    void apply_geometry (void);
    void set_geometry_from_plm_image_header (Plm_image_header *pih);
    void set_geometry_from_plm_image (Plm_image *pli);
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
