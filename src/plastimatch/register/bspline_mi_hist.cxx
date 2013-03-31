/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <stdio.h>
#include "bspline_mi_hist.h"
#include "file_util.h"
#include "string_util.h"

void
Bspline_mi_hist_set::dump_hist (int it, const std::string& prefix)
{
    double* f_hist = this->f_hist;
    double* m_hist = this->m_hist;
    double* j_hist = this->j_hist;
    plm_long i, j, v;
    FILE *fp;
    //char fn[_MAX_PATH];
    std::string fn;
    //char buf[_MAX_PATH];
    std::string buf;

    buf = string_format ("hist_fix_%02d.csv", it);
    //sprintf (buf, "hist_fix_%02d.csv", it);
    fn = prefix + buf;
    make_directory_recursive (fn.c_str());
    fp = fopen (fn.c_str(), "wb");
    if (!fp) return;
    for (plm_long i = 0; i < this->fixed.bins; i++) {
        fprintf (fp, "%u %f\n", (unsigned int) i, f_hist[i]);
    }
    fclose (fp);

    //sprintf (buf, "hist_mov_%02d.csv", it);
    buf = string_format ("hist_mov_%02d.csv", it);
    fn = prefix + buf;
    make_directory_recursive (fn.c_str());
    fp = fopen (fn.c_str(), "wb");
    if (!fp) return;
    for (i = 0; i < this->moving.bins; i++) {
        fprintf (fp, "%u %f\n", (unsigned int) i, m_hist[i]);
    }
    fclose (fp);

    //sprintf (buf, "hist_jnt_%02d.csv", it);
    buf = string_format ("hist_jnt_%02d.csv", it);
    fn = prefix + buf;
    make_directory_recursive (fn.c_str());
    fp = fopen (fn.c_str(), "wb");
    if (!fp) return;
    for (i = 0, v = 0; i < this->fixed.bins; i++) {
        for (j = 0; j < this->moving.bins; j++, v++) {
            if (j_hist[v] > 0) {
                fprintf (fp, "%u %u %u %g\n", (unsigned int) i, 
                    (unsigned int) j, (unsigned int) v, j_hist[v]);
            }
        }
    }
    fclose (fp);
}

