/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "plm_math.h"
#include "rtplan_control_pt.h"
#include "rtplan_beam.h"
#include "string_util.h"
#include "logfile.h"

Rtplan_beam::Rtplan_beam()
{
}

Rtplan_beam::~Rtplan_beam()
{
    this->clear ();
}

void
Rtplan_beam::clear()
{  
    this->name = "";
    this->description = "";
    this->final_cumulative_meterset_weight = 0.f;

    for (size_t i = 0; i < this->cplist.size(); i++) {
        delete this->cplist[i];
    }
    this->cplist.clear ();
}

Rtplan_control_pt*
Rtplan_beam::add_control_pt ()
{
    Rtplan_control_pt* new_control_pt = new Rtplan_control_pt;
    this->cplist.push_back (new_control_pt);
    return new_control_pt;
}

/* first beam isocenter position*/
#if defined (commentout)
float*
Rtplan_beam::get_isocenter_pos()
{
    float isopos[3];

    isopos[0] = 0.0;
    isopos[1] = 0.0;
    isopos[2] = 0.0;

    if (num_cp < 1)
        return isopos;

    /* Get First CP's isocenter position*/
    isopos[0] = cplist[0]->iso_pos[0];
    isopos[1] = cplist[0]->iso_pos[1];
    isopos[2] = cplist[0]->iso_pos[2];

    return isopos;
}
#endif

bool 
Rtplan_beam::check_isocenter_identical()
{
    bool bSame = true;
    if (this->cplist.size() < 1) {
        return false;
    }
    float *firstIso = this->cplist[0]->get_isocenter();

    for (size_t i = 1; i < this->cplist.size(); i++){
        float *currIso = this->cplist[i]->get_isocenter();
        if (firstIso[0] != currIso[0] || firstIso[1] != currIso[1]
            || firstIso[2] != currIso[2])
        {
            bSame = false;
            break;
        }
    }
    /* list isocenter positions */
    if (!bSame) {
        lprintf ("Warning! Isocenter positions are not same across the control points!\n");

        for (size_t i = 0; i < this->cplist.size(); i++){
            float *iso = this->cplist[i]->get_isocenter();
            lprintf ("Control point idx: %d,"
                " isocenter: %3.2f / %3.2f / %3.2f. \n",
                (int) i, iso[0], iso[1], iso[2]);
        }
    }
    return bSame;
}
