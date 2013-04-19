/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_slice_data_h_
#define _dcmtk_slice_data_h_

#include "plmbase_config.h"

#include "plm_int.h"
#include "pstring.h"
#include "volume.h"

class Dcmtk_slice_data
{
public:
    Pstring fn;
    Volume::Pointer vol;

    size_t slice_size;
    float *slice_float;
    int16_t *slice_int16;

    char slice_uid[100];
    Pstring ipp;
    Pstring iop;
    Pstring sloc;
    Pstring sthk;
};

#endif
