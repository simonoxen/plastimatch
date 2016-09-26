/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtplan_h_
#define _rtplan_h_

#include "plmbase_config.h"
#include <list>
#include <vector>

#include "direction_cosines.h"
#include "plm_int.h"
#include "rt_study_metadata.h"
#include "smart_pointer.h"

class Plm_image;
class Plm_image_header;
class Rtplan_beam;

class PLMBASE_API Rtplan {
public:
    SMART_POINTER_SUPPORT(Rtplan);
public:
    /* Structures */
    std::vector<Rtplan_beam*> beamlist;
public:
    Rtplan();
    ~Rtplan();
    void init (void);
    void clear (void);
    Rtplan_beam* add_beam (
        const std::string& beam_name,         
	int beam_id);
    void delete_beam (int index);
    Rtplan_beam* find_beam_by_id (size_t index);
    std::string get_beam_name (size_t index);
    void set_beam_name (size_t index, const std::string& name);
    void debug (void);
};

#endif
