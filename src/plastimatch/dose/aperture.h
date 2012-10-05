/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _aperture_h_
#define _aperture_h_

#include "plmdose_config.h"
#include "plm_int.h"

class Aperture_private;

class PLMDOSE_API Aperture {
public:
    Aperture ();
    ~Aperture ();
public:
    Aperture_private *d_ptr;
public:
    /*! \name Inputs */
    ///@{

    /*! \brief Get the aperture dimension, in pixels */
    const int* get_dim () const;
    /*! \brief Get the i or j aperture dimension, in pixels */
    int get_dim (int dim) const;
    /*! \brief Set the aperture dimension, in pixels */
    void set_dim (const int* dim);

    /*! \brief Get the aperture offset: the distance from the
      beam source to closest point on the aperture plane */
    double get_offset () const;
    /*! \brief Get the aperture offset: the distance from the
      beam source to closest point on the aperture plane */
    void set_offset (double offset);
    ///@}

public:
//    double ap_offset;     /* distance from beam nozzle */
    double vup[3];        /* orientation */
    double ic [2];        /* center */
//    int ires[2];          /* resolution (vox) */
    double ic_room[3];    /* loc of center (room coords) */
    double ul_room[3];    /* loc of upper left corder (room coords) */
    double incr_r[3];     /* row increment vector */
    double incr_c[3];     /* col increment vector */
    double nrm[3];        /* unit vec: normal */
    double pdn[3];        /* unit vec: down */
    double prt[3];        /* unit vec: right */
    double tmp[3];
};

#endif

