/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_save_h_
#define _dcmtk_save_h_

#include "plm_config.h"
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "pstring.h"

class Rtds;
class Volume;

typedef
struct dcmtk_slice_data
{
    Pstring fn;
    Rtds *rtds;
    Volume *vol;

    size_t slice_size;
    float *slice_float;
    int16_t *slice_int16;

    OFString date_string;
    OFString time_string;
    char study_uid[100];
    char series_uid[100];
    char for_uid[100];
    char slice_uid[100];
    Pstring ipp;
    Pstring iop;
    Pstring sloc;
    Pstring sthk;
} Dcmtk_slice_data;

void
dcmtk_rtss_save (
    const std::vector<Dcmtk_slice_data> *slice_data,
    const Rtds *rtds,
    const char *dicom_dir);
void
dcmtk_save_rtds (Rtds *rtds, const char *dicom_dir);

#endif
