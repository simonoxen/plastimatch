/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtss_polyline_set_h_
#define _rtss_polyline_set_h_

#include "plmbase_config.h"
#include <list>
#include <vector>

#include "plmsys.h"

#include "pstring.h"

class Plm_image;
class Plm_image_header;
class Rtss_structure;

#define CXT_BUFLEN 2048

class Rtss_polyline_set {
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
    Rtss_polyline_set ();
    ~Rtss_polyline_set ();
    void init (void);
    void clear (void);
    Rtss_structure* add_structure (
	const Pstring& structure_name, 
	const Pstring& color, 
	int structure_id);
    Rtss_structure* find_structure_by_id (int structure_id);
    void debug (void);
    void adjust_structure_names (void);
    void prune_empty (void);
    static Rtss_polyline_set* clone_empty (Rtss_polyline_set* cxt_out, 
	Rtss_polyline_set* cxt_in);
    void find_default_geometry (Plm_image_header *pih);
    void find_rasterization_geometry (float offset[3], 
	float spacing[3], plm_long dims[3]);
    void find_rasterization_geometry (Plm_image_header *pih);
    Pstring find_unused_structure_name (void);
    void fix_polyline_slice_numbers (void);
    void free_all_polylines (void);
    void keyholize (void);
    void set_rasterization_geometry (void);
    void set_geometry_from_plm_image_header (Plm_image_header *pih);
    void set_geometry_from_plm_image (Plm_image *pli);
};

#endif
