/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_beam_h_
#define _proton_beam_h_

class Proton_sobp;

class PLMDOSE_API Proton_Beam {
public:
    Proton_Beam ();
    ~Proton_Beam ();

    bool load (const char* fn);     /* load from file */
    bool generate ();               /* generate analytically */

    void dump (const char* fn);     /* debug: print bragg curve to file */
    void add_peak ();
    float lookup_energy (float depth);

private:
    bool load_xio (const char* fn);
    bool load_txt (const char* fn);

public:
    double src[3];                  /* beam nozzle location in space */
    double isocenter[3];            /* beam is aimed at this point */

    Proton_sobp *sobp;

//    float* d_lut;                   /* depth array (mm) */
//    float* e_lut;                   /* energy array (MeV) */

    double E0;                      /* initial proton energy (MeV) */
    double spread;                  /* beam energy sigma (MeV) */
    double dres;                    /* spatial resolution of bragg curve (mm)*/
    double dmax;                    /* maximum w.e.d. (mm) */
    int num_samples;                /* # of discrete bragg curve samples */
    double weight;
};

#endif
