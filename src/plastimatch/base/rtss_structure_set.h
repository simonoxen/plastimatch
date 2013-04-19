/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtss_structure_set_h_
#define _rtss_structure_set_h_

#include "plmbase_config.h"
#include <list>
#include <vector>

#include "plm_int.h"
#include "pstring.h"
#include "smart_pointer.h"

class Plm_image;
class Plm_image_header;
class Rtss_structure;
class Slice_index;
class Slice_list;

#define CXT_BUFLEN 2048

class PLMBASE_API Rtss_structure_set {
public:
    SMART_POINTER_SUPPORT (Rtss_structure_set);
public:
    /* Output geometry */
    int have_geometry;
    plm_long m_dim[3];
    float m_spacing[3];
    float m_offset[3];
    /* Rasterization geometry */
    plm_long rast_dim[3];
    float rast_spacing[3];
    float rast_offset[3];
    /* Structures */
    size_t num_structures;
    Rtss_structure **slist;
public:
    Rtss_structure_set ();
    ~Rtss_structure_set ();
    void init (void);
    void clear (void);
    Rtss_structure* add_structure (
	const Pstring& structure_name, 
	const Pstring& color, 
	int structure_id,
        int bit = -1);
    void delete_structure (int index);
    Rtss_structure* find_structure_by_id (int structure_id);
    std::string get_structure_name (size_t index);
    void debug (void);
    void adjust_structure_names (void);
    void prune_empty (void);
    static 
    Rtss_structure_set* clone_empty (Rtss_structure_set* cxt_out, 
	Rtss_structure_set* cxt_in);
    void find_default_geometry (Plm_image_header *pih);
    void find_rasterization_geometry (float offset[3], 
	float spacing[3], plm_long dims[3]);
    void find_rasterization_geometry (Plm_image_header *pih);
    Pstring find_unused_structure_name (void);
    void fix_polyline_slice_numbers (void);
    void apply_slice_index (const Slice_index *rdd);
    void apply_slice_list (const Slice_list *slice_list);
    void free_all_polylines (void);
    void keyholize (void);
    void set_rasterization_geometry (void);
    void set_geometry (const Plm_image_header *pih);
    void set_geometry (const Plm_image *pli);
};

#endif
