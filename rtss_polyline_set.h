/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtss_polyline_set_h_
#define _rtss_polyline_set_h_

#include "plm_config.h"
#include <list>
#include <vector>
#include "bstrwrap.h"

class Img_metadata;
class Plm_image;
class Plm_image_header;
class Rtss_structure;

#define CXT_BUFLEN 2048

class Rtss_polyline_set {
public:
    //CBString ct_study_uid;
    //CBString ct_series_uid;
    //CBString ct_fref_uid;
    //CBString study_id;
    //std::vector<CBString> ct_slice_uids;

    Img_metadata *m_demographics;

    /* Output geometry */
    int have_geometry;
    int m_dim[3];
    float m_spacing[3];
    float m_offset[3];
    /* Rasterization geometry */
    int rast_dim[3];
    float rast_spacing[3];
    float rast_offset[3];
    /* Structures */
    int num_structures;
    Rtss_structure **slist;
public:
    plastimatch1_EXPORT
    Rtss_polyline_set ();
    plastimatch1_EXPORT
    ~Rtss_polyline_set ();
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
    static Rtss_polyline_set* clone_empty (Rtss_polyline_set* cxt_out, 
	Rtss_polyline_set* cxt_in);
    plastimatch1_EXPORT
    void free_all_polylines (void);
    void fix_polyline_slice_numbers (void);
    void find_rasterization_geometry (float offset[3], 
	float spacing[3], int dims[3]);
    void find_rasterization_geometry (Plm_image_header *pih);
    void set_rasterization_geometry (void);
    void set_geometry_from_plm_image_header (Plm_image_header *pih);
    void set_geometry_from_plm_image (Plm_image *pli);
    void find_default_geometry (Plm_image_header *pih);
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
