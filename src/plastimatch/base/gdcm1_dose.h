/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm1_dose_h_
#define _gdcm1_dose_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmbase.h"
 */

#include "plmbase_config.h"

class Metadata;
class Plm_image;
class Rtss;
class Slice_index;

#if GDCM_VERSION_1
/* gdcm1_dose.cxx */
C_API bool gdcm1_dose_probe (const char *dose_fn);
C_API Plm_image* gdcm1_dose_load (
        Plm_image *pli,
        const char *dose_fn,
        const char *dicom_dir
);
C_API void gdcm1_dose_save (
        Plm_image *pli, 
        const Metadata *meta, 
        const Slice_index *rdd, 
        const char *dose_fn);

/* gdcm1_series.cxx */
C_API void gdcm1_series_test (char *dicom_dir);

/* gdcm1_rtss.cxx */
C_API bool gdcm_rtss_probe (const char *rtss_fn);
C_API void gdcm_rtss_load (
        Rtss *rtss,             /* Output: this gets loaded into */
        Slice_index *rdd,       /* Output: this gets updated too */
        Metadata *meta,         /* Output: this gets updated too */
        const char *rtss_fn    /* Input: the file that gets read */
);
C_API void gdcm_rtss_save (
        Rtss *rtss,             /* Input: this is what gets saved */
        Slice_index *rdd,       /* Input: need to look at this too */
        char *rtss_fn           /* Input: name of file to write to */
);
#endif /* GDCM_VERSION_1 */


#endif /* _gdcm1_dose_h_ */
