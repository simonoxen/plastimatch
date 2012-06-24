/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtss_structure_set_h_
#define _rtss_structure_set_h_

#include "plmbase_config.h"
#include <list>
#include <vector>

#include "pstring.h"

#include "plmbase_config.h"

class Plm_image;
class Plm_image_header;
class Rtss_structure;

#define CXT_BUFLEN 2048

class Rtss_structure_set {
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
    PLMBASE_API Rtss_structure_set ();
    PLMBASE_API ~Rtss_structure_set ();
    PLMBASE_API void init (void);
    PLMBASE_API void clear (void);
    PLMBASE_API Rtss_structure* add_structure (
	const Pstring& structure_name, 
	const Pstring& color, 
	int structure_id,
        int bit = -1);
    PLMBASE_API Rtss_structure* find_structure_by_id (int structure_id);
    PLMBASE_API void debug (void);
    PLMBASE_API void adjust_structure_names (void);
    PLMBASE_API void prune_empty (void);
    static PLMBASE_API 
    Rtss_structure_set* clone_empty (Rtss_structure_set* cxt_out, 
	Rtss_structure_set* cxt_in);
    PLMBASE_API void find_default_geometry (Plm_image_header *pih);
    PLMBASE_API void find_rasterization_geometry (float offset[3], 
	float spacing[3], plm_long dims[3]);
    PLMBASE_API void find_rasterization_geometry (Plm_image_header *pih);
    PLMBASE_API Pstring find_unused_structure_name (void);
    PLMBASE_API void fix_polyline_slice_numbers (void);
    PLMBASE_API void free_all_polylines (void);
    PLMBASE_API void keyholize (void);
    PLMBASE_API void set_rasterization_geometry (void);
    PLMBASE_API void set_geometry (const Plm_image_header *pih);
    PLMBASE_API void set_geometry (const Plm_image *pli);
};

#endif
