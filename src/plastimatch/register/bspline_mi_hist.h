/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_mi_hist_h_
#define _bspline_mi_hist_h_

#include "plmregister_config.h"
#include <string>
#include "plm_int.h"

/* -----------------------------------------------------------------------
   Types
   ----------------------------------------------------------------------- */
enum BsplineHistType {
    HIST_EQSP,
    HIST_VOPT
};

class Bspline_mi_hist {
public:
    /* Used by all histogram types */
    enum BsplineHistType type;  /* Type of histograms */
    plm_long bins;           /* # of bins in histogram  */
    float offset;               /* minimum voxel intensity */
    plm_long big_bin;             /* fullest bin index       */
    float delta;                /* bin OR key spacing   */

    /* For V-Optimal Histograms */
    plm_long keys;                /* # of keys               */
    int* key_lut;               /* bin keys lookup table   */
};

class Bspline_mi_hist_set {
public:
    void dump_hist (int it, const std::string& prefix);
public:
    Bspline_mi_hist moving;
    Bspline_mi_hist fixed;
    Bspline_mi_hist joint;    // JAS: for big_bin
    double* m_hist;
    double* f_hist;
    double* j_hist;
};

#endif
