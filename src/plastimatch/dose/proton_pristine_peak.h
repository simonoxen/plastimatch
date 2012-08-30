/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_pristine_peak_h_
#define _proton_pristine_peak_h_

class PLMDOSE_API Proton_pristine_peak {
public:
    Proton_pristine_peak ();
    Proton_pristine_peak (double E0, double spread, double dres, 
        double dmax, double weight);
    ~Proton_pristine_peak ();

    bool load (const char* fn);     /* load from file */
    bool generate ();               /* generate analytically */

    /* debug: print bragg curve to file */
    void dump (const char* fn) const;

private:
    bool load_xio (const char* fn);
    bool load_txt (const char* fn);

public:
    float* d_lut;                   /* depth array (mm) */
    float* e_lut;                   /* energy array (MeV) */

    double E0;                      /* initial proton energy (MeV) */
    double spread;                  /* beam energy sigma (MeV) */
    double dres;                    /* spatial resolution of bragg curve (mm)*/
    double dmax;                    /* maximum w.e.d. (mm) */
    double weight;

    int num_samples;                /* # of discrete bragg curve samples */
};

#endif
