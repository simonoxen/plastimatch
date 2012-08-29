/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_sobp_h_
#define _proton_sobp_h_

class Proton_pristine_peak;

class PLMDOSE_API Proton_sobp {
public:
    Proton_sobp ();
    ~Proton_sobp ();

    /* Add a pristine peak to a sobp */
    void add (const Proton_pristine_peak& pristine_peak, double weight);

    /* Save the depth dose to a file */
    void dump (const char* fn);
};

#endif
