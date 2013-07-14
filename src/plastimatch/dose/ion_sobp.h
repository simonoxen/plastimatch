/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ion_sobp_h_
#define _ion_sobp_h_

#include "plmdose_config.h"
#include "smart_pointer.h"

class Ion_pristine_peak;
class Ion_sobp_private;

class PLMDOSE_API Ion_sobp {
public:
    SMART_POINTER_SUPPORT (Ion_sobp);
    Ion_sobp_private *d_ptr;
public:
    Ion_sobp ();
    ~Ion_sobp ();

    void set_resolution (double dres, int num_samples);

    /* Add a pristine peak to a sobp */
    void add (const Ion_pristine_peak* pristine_peak);
    void add (double E0, double spread, double dres, double dmax, 
        double weight);

    /* Set the min & max depth for automatic sobp optimization */
    void set_prescription_min_max (float d_min, float d_max);

    /* Return the maximum depth in the SOBP array */
    double get_maximum_depth ();

    /* Save the depth dose to a file */
    void dump (const char* fn);

    /* Optimize, then generate sobp depth curve from prescription 
       range and modulation */
    void optimize ();

    /* Compute the sobp depth dose curve from weighted pristine peaks */
    bool generate ();

    /* Return simple depth dose result at depth */
    float lookup_energy (float depth);

};

#endif
