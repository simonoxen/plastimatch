/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _readcxt_h_
#define _readcxt_h_

#include "bstrlib.h"
#define CXT_BUFLEN 2048

typedef struct cxt_polyline Cxt_polyline;
struct cxt_polyline {
    int slice_no;
    int num_vertices;
    float* x;
    float* y;
    float* z;
};

typedef struct cxt_structure Cxt_structure;
struct cxt_structure {
    char name[CXT_BUFLEN];
    int id;
    int num_contours;
    Cxt_polyline* pslist;
};

typedef struct cxt_structure_list Cxt_structure_list;
struct cxt_structure_list {
    int dim[3];
    float spacing[3];
    float offset[3];
    int num_structures;
    Cxt_structure* slist;
    bstring series_ct_uid;
};

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void
cxt_initialize (Cxt_structure_list* structures);
plastimatch1_EXPORT
void
cxt_add_structure (Cxt_structure_list* structures, const char *structure_name, int structure_id);
plastimatch1_EXPORT
Cxt_polyline*
cxt_add_polyline (Cxt_structure* structure);
plastimatch1_EXPORT
Cxt_structure*
cxt_find_structure_by_id (Cxt_structure_list* structures, int structure_id);
plastimatch1_EXPORT
void
cxt_read (Cxt_structure_list* structures, const char* cxt_fn);
plastimatch1_EXPORT
void
cxt_write (Cxt_structure_list* structures, const char* cxt_fn);
plastimatch1_EXPORT
void
cxt_debug_structures (Cxt_structure_list* structures);
plastimatch1_EXPORT
void
cxt_destroy (Cxt_structure_list* structures);

#if defined __cplusplus
}
#endif

#endif
