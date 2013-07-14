/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ion_beam_h_
#define _ion_beam_h_

#include "plmdose_config.h"

class Ion_beam_private;
class Ion_sobp;

class PLMDOSE_API Ion_beam {
public:
    Ion_beam ();
    ~Ion_beam ();
public:
    Ion_beam_private *d_ptr;

public:
    /*! \name Inputs */
    ///@{
    /*! \brief Load PDD from XiO or txt file */
    bool load (const char* fn);

    /*! \brief Get the position of the beam source in world coordinates. */
    const double* get_source_position ();
    /*! \brief Get the x, y, or z coordinate of the beam source 
      in world coordinates. */
    double get_source_position (int dim);
    /*! \brief Set the position of the beam source in world coordinates. */
    void set_source_position (const float position[3]);
    /*! \brief Set the position of the beam source in world coordinates. */
    void set_source_position (const double position[3]);

    /*! \brief Get the position of the beam isocenter in world coordinates. */
    const double* get_isocenter_position ();
    /*! \brief Get the x, y, or z coordinate of the beam source 
      in world coordinates. */
    double get_isocenter_position (int dim);
    /*! \brief Set the position of the beam isocenter in world coordinates. */
    void set_isocenter_position (const float position[3]);
    /*! \brief Set the position of the beam isocenter in world coordinates. */
    void set_isocenter_position (const double position[3]);

    /*! \brief Get "detail" parameter of dose calculation algorithm */
    int get_detail () const;
    /*! \brief Set "detail" parameter of dose calculation algorithm */
    void set_detail (int detail);
    /*! \brief Get "flavor" parameter of dose calculation algorithm */
    char get_flavor () const;
    /*! \brief Set "flavor" parameter of dose calculation algorithm */
    void set_flavor (char flavor);

    ///@}

    /*! \name Execution */
    ///@{
    bool generate ();               /* generate analytically */
    ///@}

    /*! \name Outputs */
    ///@{
    void dump (const char* fn);     /* debug: print bragg curve to file */
    void add_peak ();
    float lookup_energy (float depth);
    ///@}

private:
    bool load_xio (const char* fn);
    bool load_txt (const char* fn);

public:
    Ion_sobp *sobp;

    double E0;                      /* initial ion energy (MeV) */
    double spread;                  /* beam energy sigma (MeV) */
    double dres;                    /* spatial resolution of bragg curve (mm)*/
    double dmax;                    /* maximum w.e.d. (mm) */
    int num_samples;                /* # of discrete bragg curve samples */
    double weight;
};

#endif
