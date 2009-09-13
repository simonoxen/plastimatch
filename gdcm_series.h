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
    void digest (void);

    gdcm::SerieHelper2 *gdcm_sh2;
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
