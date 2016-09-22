/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <limits>
#include <set>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "plm_image_header.h"
#include "plm_int.h"
#include "plm_math.h"
#include "rtplan_beam.h"
#include "rtplan.h"
#include "rt_study_metadata.h"
#include "string_util.h"

Rtplan::Rtplan()
{
    this->init ();
}

Rtplan::~Rtplan()
{
    this->clear ();
}

void
Rtplan::init(void)
{
    this->num_beams = 0;
    this->beamlist = 0;
}

void
Rtplan::clear(void)
{
    for (size_t i = 0; i < this->num_beams; i++) {
        delete (this->beamlist[i]);
    }
    free(this->beamlist);

    this->init ();
}

/* Add structure (if it doesn't already exist) */
Rtplan_beam*
Rtplan::add_beam (
    const std::string& beam_name,     
    int beam_id)
{
    Rtplan_beam* new_beam;

    new_beam = this->find_beam_by_id(beam_id);
    if (new_beam) {
        return new_beam;
    }

    this->num_beams++;
    this->beamlist = (Rtplan_beam **) realloc (this->beamlist, 
        this->num_beams * sizeof(Rtplan_beam*));
    new_beam 
	= this->beamlist[this->num_beams - 1] 
	= new Rtplan_beam;

    new_beam->name = beam_name;
    if (beam_name == "") {
        new_beam->name = "Unknown beam";
    }
    new_beam->name = string_trim (new_beam->name);
    new_beam->id = beam_id;

    return new_beam;
}

void
Rtplan::delete_beam(int index)
{
    Rtplan_beam* curr_beam = this->beamlist[index];
    delete curr_beam;

    this->beamlist[index] = this->beamlist[this->num_beams - 1];
    this->num_beams--;
}

Rtplan_beam*
Rtplan::find_beam_by_id (int beam_id)
{
    for (size_t i = 0; i < this->num_beams; i++) {
	Rtplan_beam* curr_beam;
	curr_beam = this->beamlist[i];
	if (curr_beam->id == beam_id) {
	    return curr_beam;
	}
    }
    return 0;
}

void 
Rtplan::set_beam_name (size_t index, const std::string& name)
{
    if (index < this->num_beams) {
        this->beamlist[index]->name = name.c_str();
    }
}

std::string
Rtplan::get_beam_name(size_t index)
{
    if (index < this->num_beams) {
        return std::string (this->beamlist[index]->name.c_str());
    } else {
        return "";
    }
}

void
Rtplan::debug(void)
{   
}
