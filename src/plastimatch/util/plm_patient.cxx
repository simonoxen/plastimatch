/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <list>
#include <stdio.h>
#include <stdlib.h>

#include "metadata.h"
#include "plm_patient.h"
#include "plm_series.h"

class Plm_patient_private {
public:
    Metadata *meta;
    std::list<Plm_series> series_list;
    
public:
    Plm_patient_private () {
        meta = new Metadata;
    }
    ~Plm_patient_private () {
        delete meta;
    }
};

Plm_patient::Plm_patient ()
{
    d_ptr = new Plm_patient_private;
}

Plm_patient::~Plm_patient ()
{
    delete d_ptr;
}

void
Plm_patient::load_rdd (const char* rdd)
{
}

void
Plm_patient::debug () const
{
}
