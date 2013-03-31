/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_mi_hist_h_
#define _bspline_mi_hist_h_

#include "plmregister_config.h"
#include <string>
#include "plm_int.h"

/* Maximum # of bins for a vopt histogram */
#define VOPT_RES 1000

class Volume;

/* -----------------------------------------------------------------------
   Types
   ----------------------------------------------------------------------- */
enum Bspline_mi_hist_type {
    HIST_EQSP,
    HIST_VOPT
};

class Bspline_mi_hist {
public:
    Bspline_mi_hist (
        Bspline_mi_hist_type type = HIST_EQSP, 
        plm_long bins = 32);
    ~Bspline_mi_hist ();
public:
    /* Used by all histogram types */
    enum Bspline_mi_hist_type type;   /* Type of histograms */
    plm_long bins;                    /* # of bins in histogram  */
    float offset;                     /* minimum voxel intensity */
    plm_long big_bin;                 /* fullest bin index       */
    float delta;                      /* bin OR key spacing   */

    /* For V-Optimal Histograms */
    plm_long keys;                    /* # of keys               */
    int* key_lut;                     /* bin keys lookup table   */
};

class Bspline_mi_hist_set {
public:
    Bspline_mi_hist_set ();
    Bspline_mi_hist_set (
        Bspline_mi_hist_type type,
        plm_long fixed_bins,
        plm_long moving_bins);
    ~Bspline_mi_hist_set ();
public:
    void initialize (Volume *fixed, Volume *moving);
    void dump_hist (int it, const std::string& prefix);
public:
    Bspline_mi_hist moving;
    Bspline_mi_hist fixed;
    Bspline_mi_hist joint;    // JAS: for big_bin
    double* m_hist;
    double* f_hist;
    double* j_hist;
protected:
    void allocate ();
};

#endif
