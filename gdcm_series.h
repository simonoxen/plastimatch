/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm_series_h_
#define _gdcm_series_h_

#include "plm_config.h"
#include <vector>
#include <map>
#include "gdcm_series_helper_2.h"

class Gdcm_series 
{
public:
    Gdcm_series ();
    ~Gdcm_series ();

    void load (char *dicom_dir);
    void get_best_ct (void);

    gdcm::SerieHelper2 *m_gsh2;

    int m_have_ct;
    int m_dim[3];
    double m_origin[3];
    double m_spacing[3];
};

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void
gdcm_series_test (char *dicom_dir);

#if defined __cplusplus
}
#endif

#endif
