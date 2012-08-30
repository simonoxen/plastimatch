/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_sobp_h_
#define _proton_sobp_h_

class Proton_pristine_peak;
class Proton_sobp_private;

class PLMDOSE_API Proton_sobp {
public:
    Proton_sobp ();
    ~Proton_sobp ();

    void set_resolution (double dres, int num_samples);

    /* Add a pristine peak to a sobp */
    void add (const Proton_pristine_peak* pristine_peak);
    void add (double E0, double spread, double dres, double dmax, 
        double weight);

    /* Save the depth dose to a file */
    void dump (const char* fn);

    /* Compute the sobp depth dose curve from weighted pristine peaks */
    bool generate ();

    /* Return simple pdd result at depth */
    float lookup_energy (float depth);

public:
    Proton_sobp_private *d_ptr;
};

#endif
