/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_beam_h_
#define _proton_beam_h_

class Proton_Beam {
public:
    Proton_Beam ();
    ~Proton_Beam ();

    void load (const char* fn);     /* load from file */
    void generate ();               /* generate analytically */

private:
    void load_xio (const char* fn);
    void load_txt (const char* fn);

public:
    double src[3];                  /* beam nozzle location in space */
    double isocenter[3];            /* beam is aimed at this point */

    float* d_lut;                   /* depth array (mm) */
    float* e_lut;                   /* energy array (MeV) */

    double E0;                      /* initial proton energy (MeV) */
    double spread;                  /* beam energy sigma (MeV) */
    double dmax;                    /* maximum w.e.d. (mm) */
    double dres;                    /* spatial resolution of bragg curve (mm)*/
    int num_samples;                /* # of discrete bragg curve samples */
};

#endif
