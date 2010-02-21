/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cxt_h_
#define _cxt_h_

#include "plm_config.h"
#include "bstrlib.h"
#define CXT_BUFLEN 2048
#include "plm_image.h"
#include "plm_image_header.h"

typedef struct cxt_polyline Cxt_polyline;
struct cxt_polyline {
    int slice_no;
    bstring ct_slice_uid;
    int num_vertices;
    float* x;
    float* y;
    float* z;
};

typedef struct cxt_structure Cxt_structure;
struct cxt_structure {
    char name[CXT_BUFLEN];
    bstring color;
    int id;
    int num_contours;
    Cxt_polyline* pslist;
};

typedef struct cxt_structure_list Cxt_structure_list;
struct cxt_structure_list {
    bstring ct_study_uid;
    bstring ct_series_uid;
    bstring ct_fref_uid;
    bstring patient_name;
    bstring patient_id;
    bstring patient_sex;
    bstring study_id;
    int have_geometry;
    int dim[3];
    float spacing[3];
    float offset[3];
    int num_structures;
    Cxt_structure* slist;
};

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
Cxt_structure_list*
cxt_create (void);
plastimatch1_EXPORT
void
cxt_init (Cxt_structure_list* structures);
plastimatch1_EXPORT
void
cxt_add_structure (Cxt_structure_list* structures, const char *structure_name,
		   bstring color, int structure_id);
plastimatch1_EXPORT
Cxt_polyline*
cxt_add_polyline (Cxt_structure* structure);
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
cxt_destroy (Cxt_structure_list* structures);
plastimatch1_EXPORT
void
cxt_prune_empty (Cxt_structure_list* structures);
plastimatch1_EXPORT
void
cxt_apply_geometry (Cxt_structure_list* structures);
plastimatch1_EXPORT
void
cxt_set_geometry_from_plm_image_header (
    Cxt_structure_list* cxt, 
    PlmImageHeader *pih
);
plastimatch1_EXPORT
void
cxt_set_geometry_from_plm_image (
    Cxt_structure_list* structures,
    PlmImage *pli
);

#if defined __cplusplus
}
#endif

#endif
